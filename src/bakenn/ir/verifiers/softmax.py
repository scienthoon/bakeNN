from __future__ import annotations

from bakenn.errors import GraphValidationError
from bakenn.ir.graph import QuantizedGraph
from bakenn.ir.ops.softmax import SoftmaxOp
from bakenn.ir.types import DType, Layout, PerTensorQParams
from bakenn.ir.verify import verify_op


SOFTMAX_OUTPUT_SCALE = 1.0 / 256.0
SOFTMAX_OUTPUT_ZERO_POINT = -128
UINT32_MAX = (1 << 32) - 1
Q15_ONE = (1 << 15) - 1


@verify_op.register
def _verify_softmax(op: SoftmaxOp, graph: QuantizedGraph) -> None:
    input_type = graph.values[op.input]
    output_type = graph.values[op.output]
    prefix = f"{op.name}: "
    if input_type.dtype is not DType.INT8 or output_type.dtype is not DType.INT8:
        raise GraphValidationError(prefix + "Softmax tensors must be int8")
    if input_type.layout is not Layout.NC or output_type.layout is not Layout.NC:
        raise GraphValidationError(prefix + "Softmax tensors must use NC layout")
    if len(input_type.shape) != 2 or input_type.shape[0] != 1:
        raise GraphValidationError(prefix + "P0 Softmax requires rank-two static batch-one input")
    if output_type.shape != input_type.shape:
        raise GraphValidationError(prefix + "Softmax output shape must equal input shape")
    if not isinstance(input_type.qparams, PerTensorQParams) or not isinstance(
        output_type.qparams, PerTensorQParams
    ):
        raise GraphValidationError(prefix + "Softmax tensors require per-tensor qparams")
    if (
        output_type.qparams.scale != SOFTMAX_OUTPUT_SCALE
        or output_type.qparams.zero_point != SOFTMAX_OUTPUT_ZERO_POINT
    ):
        raise GraphValidationError(
            prefix + "output qparams must be scale=1/256 and zero_point=-128"
        )
    class_count = input_type.shape[1]
    if class_count * Q15_ONE > UINT32_MAX:
        raise GraphValidationError(prefix + "Q15 LUT sum may overflow uint32")


__all__ = [
    "Q15_ONE",
    "SOFTMAX_OUTPUT_SCALE",
    "SOFTMAX_OUTPUT_ZERO_POINT",
    "UINT32_MAX",
]
