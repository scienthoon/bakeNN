from __future__ import annotations

import math
import numpy as np

from bakenn.errors import GraphValidationError
from bakenn.ir.graph import QuantizedGraph
from bakenn.ir.ops.sequence import AveragePool1DOp, Conv1DOp, MaxPool1DOp
from bakenn.ir.types import DType, Layout, PerAxisQParams, PerTensorQParams, TARGET_SIZE_MAX, normalize_scale_float32
from bakenn.ir.verify import verify_op
from bakenn.quantization.fixedpoint import INT32_MAX


def _fail(op: object, reason: str) -> None:
    raise GraphValidationError(f"{getattr(op, 'name')}: {reason}")


@verify_op.register
def _verify_conv1d(op: Conv1DOp, graph: QuantizedGraph) -> None:
    x, weight, bias, output = (graph.values[name] for name in (op.input, op.weight, op.bias, op.output))
    if op.weight not in graph.constants or op.bias not in graph.constants:
        _fail(op, "Conv1D weight and bias must be constants")
    if x.dtype is not DType.INT8 or output.dtype is not DType.INT8:
        _fail(op, "Conv1D activations must be int8")
    if x.layout is not Layout.NLC or output.layout is not Layout.NLC:
        _fail(op, "Conv1D activations must use NLC layout")
    if len(x.shape) != 3 or len(output.shape) != 3 or x.shape[0] != 1 or output.shape[0] != 1:
        _fail(op, "Conv1D requires rank-three static batch size one")
    if not isinstance(x.qparams, PerTensorQParams) or not isinstance(output.qparams, PerTensorQParams):
        _fail(op, "Conv1D activations require per-tensor qparams")
    if weight.dtype is not DType.INT8 or weight.layout is not Layout.OWI or len(weight.shape) != 3:
        _fail(op, "Conv1D weights must be rank-three OWI int8")
    output_channels, kernel, group_input_channels = weight.shape
    if x.shape[2] % op.groups or output_channels % op.groups:
        _fail(op, "Conv1D groups must divide input and output channels")
    if group_input_channels != x.shape[2] // op.groups:
        _fail(op, "Conv1D OWI input channels must equal input channels / groups")
    if bias.dtype is not DType.INT32 or bias.layout is not Layout.C or bias.shape != (output_channels,):
        _fail(op, "Conv1D bias must be one int32 C value per output channel")
    if not isinstance(weight.qparams, PerAxisQParams) or weight.qparams.axis != 0:
        _fail(op, "Conv1D weights require per-output-channel qparams")
    if any(weight.qparams.zero_points) or np.any(graph.constants[op.weight] == -128):
        _fail(op, "Conv1D symmetric weights require zero_point zero and range [-127,127]")
    if not isinstance(bias.qparams, PerAxisQParams) or bias.qparams.axis != 0 or any(bias.qparams.zero_points):
        _fail(op, "Conv1D bias requires zero-point-zero per-output-channel qparams")
    expected_bias_scales = tuple(normalize_scale_float32(x.qparams.scale * scale) for scale in weight.qparams.scales)
    if len(expected_bias_scales) != output_channels or len(bias.qparams.scales) != output_channels or any(
        not math.isclose(a, b, rel_tol=1e-12) for a, b in zip(bias.qparams.scales, expected_bias_scales)
    ):
        _fail(op, "Conv1D bias scales must equal input_scale * weight_scale[channel]")
    effective = op.dilation * (kernel - 1) + 1
    numerator = x.shape[1] + sum(op.padding) - effective
    expected_length = numerator // op.stride + 1 if numerator >= 0 else 0
    if output.shape != (1, expected_length, output_channels) or expected_length <= 0:
        _fail(op, f"Conv1D output shape must be (1, {expected_length}, {output_channels})")
    if max(op.stride, op.dilation, op.groups, *op.padding) > TARGET_SIZE_MAX:
        _fail(op, "Conv1D parameters exceed target ABI")


def _verify_pool(op: AveragePool1DOp | MaxPool1DOp, graph: QuantizedGraph) -> None:
    x, output = graph.values[op.input], graph.values[op.output]
    if x.dtype is not DType.INT8 or output.dtype is not DType.INT8 or x.layout is not Layout.NLC or output.layout is not Layout.NLC:
        _fail(op, "Pool1D requires int8 NLC activations")
    if len(x.shape) != 3 or len(output.shape) != 3 or x.shape[0] != 1 or output.shape[0] != 1:
        _fail(op, "Pool1D requires rank-three static batch size one")
    if not isinstance(x.qparams, PerTensorQParams) or x.qparams != output.qparams:
        _fail(op, "Pool1D must preserve per-tensor qparams")
    expected_length = (x.shape[1] + sum(op.padding) - op.kernel) // op.stride + 1
    if expected_length <= 0 or output.shape != (1, expected_length, x.shape[2]):
        _fail(op, f"Pool1D output shape must be (1, {expected_length}, {x.shape[2]})")
    left, _ = op.padding
    for position in (0, expected_length - 1):
        start = position * op.stride - left
        if max(0, min(start + op.kernel, x.shape[1]) - max(start, 0)) <= 0:
            _fail(op, "every Pool1D window must contain an input element")
    if isinstance(op, AveragePool1DOp) and not isinstance(op, MaxPool1DOp):
        max_abs = max(abs(-128 - x.qparams.zero_point), abs(127 - x.qparams.zero_point))
        if max_abs * op.kernel > INT32_MAX:
            _fail(op, "AveragePool1D accumulator can overflow int32")


@verify_op.register
def _verify_average_pool1d(op: AveragePool1DOp, graph: QuantizedGraph) -> None: _verify_pool(op, graph)


@verify_op.register
def _verify_max_pool1d(op: MaxPool1DOp, graph: QuantizedGraph) -> None: _verify_pool(op, graph)


__all__: list[str] = []
