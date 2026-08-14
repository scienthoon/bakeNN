from __future__ import annotations

from math import prod

from bakenn.errors import GraphValidationError
from bakenn.ir.graph import QuantizedGraph
from bakenn.ir.ops.shape import ConcatenateOp, FlattenOp, ReshapeOp, SliceOp
from bakenn.ir.types import DType, Layout, PerTensorQParams, TensorType
from bakenn.ir.verify import verify_op


def _fail(op: object, message: str) -> None:
    raise GraphValidationError(f"{getattr(op, 'name', '<shape>')}: {message}")


def _canonical_activation(tensor: TensorType) -> bool:
    return (
        tensor.layout is Layout.NC
        and len(tensor.shape) == 2
        and tensor.shape[0] == 1
    ) or (
        tensor.layout is Layout.NLC
        and len(tensor.shape) == 3
        and tensor.shape[0] == 1
    ) or (
        tensor.layout is Layout.NHWC
        and len(tensor.shape) == 4
        and tensor.shape[0] == 1
    )


def _verify_view(op: ReshapeOp | FlattenOp, graph: QuantizedGraph) -> None:
    input_type = graph.values[op.input]
    output_type = graph.values[op.output]
    if input_type.dtype is not DType.INT8 or output_type.dtype is not DType.INT8:
        _fail(op, "view activations must be int8")
    if not _canonical_activation(input_type) or not _canonical_activation(output_type):
        _fail(op, "view tensors must use canonical batch-one NC, NLC, or NHWC layout")
    if not isinstance(input_type.qparams, PerTensorQParams) or not isinstance(
        output_type.qparams, PerTensorQParams
    ):
        _fail(op, "view activations require per-tensor qparams")
    if input_type.qparams != output_type.qparams:
        _fail(op, "view operations must preserve qparams")
    if input_type.numel != output_type.numel:
        _fail(op, "view operations must preserve the number of elements")
    if isinstance(op, FlattenOp):
        if input_type.layout not in (Layout.NHWC, Layout.NLC) or output_type.layout is not Layout.NC:
            _fail(op, "Flatten requires NHWC or NLC input and NC output")
        if output_type.shape != (1, prod(input_type.shape[1:])):
            _fail(op, "Flatten output must contain all non-batch input elements")


@verify_op.register
def _verify_reshape(op: ReshapeOp, graph: QuantizedGraph) -> None:
    _verify_view(op, graph)


@verify_op.register
def _verify_flatten(op: FlattenOp, graph: QuantizedGraph) -> None:
    _verify_view(op, graph)


@verify_op.register
def _verify_slice(op: SliceOp, graph: QuantizedGraph) -> None:
    input_type = graph.values[op.input]
    output_type = graph.values[op.output]
    if not _canonical_activation(input_type) or not _canonical_activation(output_type):
        _fail(op, "Slice tensors must use canonical batch-one NC, NLC, or NHWC layout")
    if input_type.dtype is not DType.INT8 or output_type.dtype is not DType.INT8:
        _fail(op, "Slice tensors must be int8")
    if input_type.layout is not output_type.layout or input_type.qparams != output_type.qparams:
        _fail(op, "Slice must preserve layout and qparams")
    rank = len(input_type.shape)
    if op.axis >= rank:
        _fail(op, f"Slice axis {op.axis} is outside rank {rank}")
    if op.stop > input_type.shape[op.axis]:
        _fail(op, "Slice stop exceeds the input dimension")
    expected = list(input_type.shape)
    expected[op.axis] = (op.stop - op.start + op.step - 1) // op.step
    if output_type.shape != tuple(expected):
        _fail(op, f"Slice output shape must be {tuple(expected)}")


@verify_op.register
def _verify_concatenate(op: ConcatenateOp, graph: QuantizedGraph) -> None:
    inputs = [graph.values[name] for name in op.input_names]
    output = graph.values[op.output]
    rank = len(inputs[0].shape)
    if rank not in (2, 3, 4) or not all(len(item.shape) == rank for item in (*inputs, output)):
        _fail(op, "Concatenate supports equal-rank NC, NLC, or NHWC tensors")
    expected_layout = Layout.NC if rank == 2 else Layout.NLC if rank == 3 else Layout.NHWC
    if not all(item.layout is expected_layout for item in (*inputs, output)):
        _fail(op, f"all tensors must use {expected_layout.value} layout")
    if not all(item.shape[0] == 1 for item in (*inputs, output)):
        _fail(op, "P0 Concatenate requires static batch size one")
    if not all(item.dtype is DType.INT8 for item in (*inputs, output)):
        _fail(op, "Concatenate tensors must be int8")
    if not all(isinstance(item.qparams, PerTensorQParams) for item in (*inputs, output)):
        _fail(op, "Concatenate tensors require per-tensor qparams")
    if not all(item.qparams == output.qparams for item in inputs):
        _fail(op, "legalized Concatenate requires identical input and output qparams")

    if not -rank <= op.axis < rank:
        _fail(op, f"axis {op.axis} is outside rank {rank}")
    axis = op.axis % rank
    if axis == 0:
        _fail(op, "Concatenate may not change the batch dimension")
    expected_shape = list(inputs[0].shape)
    expected_shape[axis] = sum(item.shape[axis] for item in inputs)
    for item in inputs[1:]:
        if any(item.shape[index] != inputs[0].shape[index] for index in range(rank) if index != axis):
            _fail(op, "all non-concatenated dimensions must match")
    if output.shape != tuple(expected_shape):
        _fail(op, f"output shape must be {tuple(expected_shape)}")


__all__: list[str] = []
