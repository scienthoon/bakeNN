from __future__ import annotations

from math import prod

from bakenn.errors import CompileError
from bakenn.ir.graph import QuantizedGraph
from bakenn.ir.ops.tensor import Pad2DOp, ReduceMeanOp
from bakenn.ir.types import PerTensorQParams
from bakenn.plan.lower import lower_op
from bakenn.plan.steps.tensor import Pad2DStep, ReduceMeanStep
from bakenn.quantization.fixedpoint import INT32_MAX, quantize_multiplier


@lower_op.register
def _lower_pad2d(op: Pad2DOp, graph: QuantizedGraph) -> Pad2DStep:
    del graph
    return Pad2DStep(op.name, op.input, op.output, op.padding)


@lower_op.register
def _lower_reduce_mean(op: ReduceMeanOp, graph: QuantizedGraph) -> ReduceMeanStep:
    input_type = graph.values[op.input]
    output_type = graph.values[op.output]
    input_qparams = input_type.qparams
    output_qparams = output_type.qparams
    assert isinstance(input_qparams, PerTensorQParams)
    assert isinstance(output_qparams, PerTensorQParams)
    rank = len(input_type.shape)
    axes = tuple(axis % rank for axis in op.axes)
    position_count = prod(input_type.shape[axis] for axis in axes)
    channels = input_type.shape[-1]
    max_abs = max(
        abs(-128 - input_qparams.zero_point),
        abs(127 - input_qparams.zero_point),
    )
    bound = position_count * max_abs
    multiplier, shift = quantize_multiplier(
        input_qparams.scale / (output_qparams.scale * position_count)
    )
    if shift > 0 and bound * (1 << shift) > INT32_MAX:
        raise CompileError(f"{op.name}: ReduceMean requantization left shift is not int32-safe")
    return ReduceMeanStep(
        op.name,
        op.input,
        op.output,
        position_count,
        channels,
        multiplier,
        shift,
        bound,
    )


__all__: list[str] = []
