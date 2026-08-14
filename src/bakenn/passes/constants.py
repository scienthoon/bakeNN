from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

from bakenn.errors import CompileError
from bakenn.ir.graph import QuantizedGraph
from bakenn.ir.op import LinearOp
from bakenn.ir.ops.conv import Conv2DOp, DepthwiseConv2DOp
from bakenn.ir.types import PerAxisQParams, PerTensorQParams, TensorType
from bakenn.ir.verify import verify_graph
from bakenn.quantization.fixedpoint import (
    INT32_MAX,
    multiply_by_quantized_multiplier,
    quantize_multiplier,
)


_CHANNEL_OPS = (LinearOp, Conv2DOp, DepthwiseConv2DOp)


@dataclass(frozen=True)
class ConstantChannel:
    """Exact portable-v1 result of one input-independent output channel.

    This is an analysis result, not a claim that the channel was removed from
    the graph.  P0 has no typed mixed constant/dynamic channel construction,
    so the original compute operation remains the safe representation.
    """

    op_name: str
    op_kind: str
    output: str
    channel: int
    bias_code: int
    multiplier: int
    shift: int
    output_code: int


def _channel_is_zero(weight: np.ndarray, channel: int, axis: int) -> bool:
    if axis == 0:
        return not np.any(weight[channel])
    if axis == 2:
        return not np.any(weight[:, :, channel])
    raise AssertionError("unsupported output-channel axis")


def analyze_constant_channels(graph: QuantizedGraph) -> tuple[ConstantChannel, ...]:
    """Prove the exact int8 result of every zero-weight compute channel.

    Conv2D, DepthwiseConv2D, and Linear subtract the input zero point before
    accumulation.  A channel whose quantized weights are all zero therefore
    has accumulator ``bias[channel]`` for every input and padding position.
    This function applies the same Q31 multiplier, rounding, output zero point,
    and fused clamp as the portable-v1 reference and generated C kernels.

    The function is deliberately fail-closed when an otherwise valid graph is
    not lowerable under portable-v1 shift/overflow rules.  It does not invent a
    weight scale or recover an FP32 bias that was discarded before this IR.
    """

    if not isinstance(graph, QuantizedGraph):
        raise CompileError("constant-channel analysis requires a QuantizedGraph")
    verify_graph(graph)

    result: list[ConstantChannel] = []
    for op in graph.ops:
        if not isinstance(op, _CHANNEL_OPS):
            continue

        input_qparams = graph.values[op.input].qparams
        weight_qparams = graph.values[op.weight].qparams
        output_qparams = graph.values[op.output].qparams
        # verify_graph establishes these contracts.  Keeping explicit checks
        # here prevents an import-order or future verifier regression from
        # turning the analysis into an unchecked assertion in optimized mode.
        if not isinstance(input_qparams, PerTensorQParams):
            raise CompileError(f"{op.name}: constant-channel input qparams are not per-tensor")
        if not isinstance(weight_qparams, PerAxisQParams):
            raise CompileError(f"{op.name}: constant-channel weight qparams are not per-axis")
        if not isinstance(output_qparams, PerTensorQParams):
            raise CompileError(f"{op.name}: constant-channel output qparams are not per-tensor")

        weight = graph.constants[op.weight]
        bias = graph.constants[op.bias]
        channel_axis = 2 if isinstance(op, DepthwiseConv2DOp) else 0
        output_channels = weight.shape[channel_axis]
        for channel in range(output_channels):
            if not _channel_is_zero(weight, channel, channel_axis):
                continue

            bias_code = int(bias[channel])
            # The plan's symmetric accumulator-bound proof intentionally
            # excludes INT32_MIN because abs(INT32_MIN) is not representable
            # as int32.  Match that lowering contract exactly.
            if abs(bias_code) > INT32_MAX:
                raise CompileError(
                    f"{op.name} channel {channel}: constant-channel accumulator "
                    "bound exceeds int32"
                )
            real_multiplier = (
                input_qparams.scale
                * weight_qparams.scales[channel]
                / output_qparams.scale
            )
            try:
                multiplier, shift = quantize_multiplier(real_multiplier)
            except CompileError as error:
                raise CompileError(
                    f"{op.name} channel {channel}: constant channel is not lowerable: {error}"
                ) from error
            if shift > 0 and abs(bias_code) * (1 << shift) > INT32_MAX:
                raise CompileError(
                    f"{op.name} channel {channel}: constant-channel requantization "
                    "left shift is not int32-safe"
                )
            try:
                requantized = multiply_by_quantized_multiplier(
                    bias_code, multiplier, shift
                )
            except OverflowError as error:
                raise CompileError(
                    f"{op.name} channel {channel}: constant-channel requantization "
                    "is not int32-safe"
                ) from error
            shifted = requantized + output_qparams.zero_point
            output_code = min(op.activation_max, max(op.activation_min, shifted))
            result.append(
                ConstantChannel(
                    op_name=op.name,
                    op_kind=type(op).__name__,
                    output=op.output,
                    channel=channel,
                    bias_code=bias_code,
                    multiplier=multiplier,
                    shift=shift,
                    output_code=output_code,
                )
            )
    return tuple(result)


def _constant_key(tensor_type: TensorType, value: np.ndarray) -> tuple[object, ...]:
    return (
        tensor_type,
        value.dtype.str,
        tuple(int(dimension) for dimension in value.shape),
        value.tobytes(order="C"),
    )


def _rewrite_constant_operands(op: object, replacements: dict[str, str]) -> object:
    operands = set(op.inputs)  # type: ignore[attr-defined]
    changes: dict[str, object] = {}
    for field in fields(op):
        # An operation name is metadata, even if a caller happened to choose
        # the same spelling as one of its constant operands.
        if field.name == "name":
            continue
        value = getattr(op, field.name)
        if isinstance(value, str) and value in operands and value in replacements:
            changes[field.name] = replacements[value]
        elif isinstance(value, tuple):
            rewritten = tuple(
                replacements.get(item, item)
                if isinstance(item, str) and item in operands
                else item
                for item in value
            )
            if rewritten != value:
                changes[field.name] = rewritten
    return replace(op, **changes) if changes else op


def deduplicate_constants(graph: QuantizedGraph) -> QuantizedGraph:
    """Merge byte-identical constants with identical typed semantics.

    Representatives are selected by lexicographic value name, independently
    of mapping insertion order.  Type, shape, dtype, qparams, layout, and bytes
    must all match; numerically equal constants in different quantization
    domains are intentionally not merged.  Existing tied constants (one name
    consumed by multiple operations) are already canonical and remain so.
    """

    if not isinstance(graph, QuantizedGraph):
        raise CompileError("constant deduplication requires a QuantizedGraph")
    verify_graph(graph)

    representative_by_key: dict[tuple[object, ...], str] = {}
    replacements: dict[str, str] = {}
    for name in sorted(graph.constants):
        key = _constant_key(graph.values[name], graph.constants[name])
        representative = representative_by_key.setdefault(key, name)
        if representative != name:
            replacements[name] = representative

    operations = tuple(
        _rewrite_constant_operands(op, replacements) for op in graph.ops
    )
    values = {
        name: tensor_type
        for name, tensor_type in graph.values.items()
        if name not in replacements
    }
    constants = {
        name: graph.constants[name]
        for name in sorted(graph.constants)
        if name not in replacements
    }
    result = QuantizedGraph(
        name=graph.name,
        values=values,
        constants=constants,
        ops=operations,  # type: ignore[arg-type]
        inputs=graph.inputs,
        outputs=graph.outputs,
        arithmetic_profile=graph.arithmetic_profile,
    )
    verify_graph(result)
    return result


__all__ = ["ConstantChannel", "analyze_constant_channels", "deduplicate_constants"]
