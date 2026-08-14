from __future__ import annotations

from bakenn.errors import GraphValidationError
from bakenn.ir.graph import QuantizedGraph
from bakenn.ir.ops.activation import HardSigmoidOp, HardSwishOp, SiLUOp, SigmoidOp
from bakenn.ir.types import DType, Layout, PerTensorQParams
from bakenn.ir.verify import verify_op


def _verify(op: SigmoidOp | HardSigmoidOp | HardSwishOp | SiLUOp, graph: QuantizedGraph) -> None:
    input_type = graph.values[op.input]
    output_type = graph.values[op.output]
    if input_type.dtype is not DType.INT8 or output_type.dtype is not DType.INT8:
        raise GraphValidationError(f"{op.name}: nonlinear activations require int8 tensors")
    if input_type.shape != output_type.shape or input_type.layout is not output_type.layout:
        raise GraphValidationError(f"{op.name}: nonlinear activations preserve shape and layout")
    if input_type.layout not in (Layout.NHWC, Layout.NC, Layout.NLC):
        raise GraphValidationError(f"{op.name}: unsupported activation layout")
    if input_type.shape[0] != 1:
        raise GraphValidationError(f"{op.name}: static batch size one is required")
    if not isinstance(input_type.qparams, PerTensorQParams) or not isinstance(
        output_type.qparams, PerTensorQParams
    ):
        raise GraphValidationError(f"{op.name}: per-tensor qparams are required")
    if isinstance(op, (SigmoidOp, HardSigmoidOp)) and output_type.qparams != PerTensorQParams(1.0 / 256.0, -128):
        raise GraphValidationError(
            f"{op.name}: Sigmoid output qparams must be scale=1/256, zero_point=-128"
        )


@verify_op.register
def _verify_sigmoid(op: SigmoidOp, graph: QuantizedGraph) -> None:
    _verify(op, graph)


@verify_op.register
def _verify_hardswish(op: HardSwishOp, graph: QuantizedGraph) -> None:
    _verify(op, graph)


@verify_op.register
def _verify_hardsigmoid(op: HardSigmoidOp, graph: QuantizedGraph) -> None:
    _verify(op, graph)


@verify_op.register
def _verify_silu(op: SiLUOp, graph: QuantizedGraph) -> None:
    _verify(op, graph)


__all__: list[str] = []
