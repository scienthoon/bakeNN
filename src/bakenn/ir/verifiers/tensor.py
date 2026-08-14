from __future__ import annotations

from math import prod

from bakenn.errors import GraphValidationError
from bakenn.ir.graph import QuantizedGraph
from bakenn.ir.ops.tensor import Pad2DOp, ReduceMeanOp
from bakenn.ir.types import DType, Layout, PerTensorQParams, TARGET_SIZE_MAX
from bakenn.ir.verify import verify_op


def _activation_pair(op: object, graph: QuantizedGraph):  # type: ignore[no-untyped-def]
    input_type = graph.values[getattr(op, "input")]
    output_type = graph.values[getattr(op, "output")]
    if input_type.dtype is not DType.INT8 or output_type.dtype is not DType.INT8:
        raise GraphValidationError(f"{getattr(op, 'name')}: int8 activations are required")
    if input_type.shape[0] != 1 or output_type.shape[0] != 1:
        raise GraphValidationError(f"{getattr(op, 'name')}: static batch size one is required")
    if not isinstance(input_type.qparams, PerTensorQParams) or not isinstance(
        output_type.qparams, PerTensorQParams
    ):
        raise GraphValidationError(f"{getattr(op, 'name')}: per-tensor qparams are required")
    return input_type, output_type


@verify_op.register
def _verify_pad2d(op: Pad2DOp, graph: QuantizedGraph) -> None:
    input_type, output_type = _activation_pair(op, graph)
    if input_type.layout is not Layout.NHWC or output_type.layout is not Layout.NHWC:
        raise GraphValidationError(f"{op.name}: Pad2D requires NHWC layout")
    if len(input_type.shape) != 4 or len(output_type.shape) != 4:
        raise GraphValidationError(f"{op.name}: Pad2D requires rank-four activations")
    if input_type.qparams != output_type.qparams:
        raise GraphValidationError(f"{op.name}: Pad2D must preserve qparams")
    top, bottom, left, right = op.padding
    expected = (
        1,
        input_type.shape[1] + top + bottom,
        input_type.shape[2] + left + right,
        input_type.shape[3],
    )
    if output_type.shape != expected:
        raise GraphValidationError(f"{op.name}: Pad2D output shape must be {expected}")
    if any(value > TARGET_SIZE_MAX for value in op.padding):
        raise GraphValidationError(f"{op.name}: padding exceeds target ABI")


@verify_op.register
def _verify_reduce_mean(op: ReduceMeanOp, graph: QuantizedGraph) -> None:
    input_type, output_type = _activation_pair(op, graph)
    if not op.keepdims:
        raise GraphValidationError(f"{op.name}: ReduceMean v1 requires keepdims=True")
    rank = len(input_type.shape)
    axes = tuple(sorted({axis % rank for axis in op.axes}))
    if len(axes) != len(op.axes):
        raise GraphValidationError(f"{op.name}: ReduceMean axes must be unique and in range")
    if input_type.layout is Layout.NHWC:
        expected_axes = (1, 2)
    elif input_type.layout is Layout.NLC:
        expected_axes = (1,)
    else:
        raise GraphValidationError(f"{op.name}: ReduceMean supports NHWC spatial or NLC time axes")
    if axes != expected_axes or output_type.layout is not input_type.layout:
        raise GraphValidationError(f"{op.name}: unsupported ReduceMean axes/layout combination")
    expected_shape = tuple(1 if index in axes else value for index, value in enumerate(input_type.shape))
    if output_type.shape != expected_shape:
        raise GraphValidationError(f"{op.name}: ReduceMean output shape must be {expected_shape}")
    count = prod(input_type.shape[index] for index in axes)
    qparams = input_type.qparams
    assert isinstance(qparams, PerTensorQParams)
    max_abs = max(abs(-128 - qparams.zero_point), abs(127 - qparams.zero_point))
    if count * max_abs > (1 << 31) - 1:
        raise GraphValidationError(f"{op.name}: ReduceMean centered sum can overflow int32")


__all__: list[str] = []
