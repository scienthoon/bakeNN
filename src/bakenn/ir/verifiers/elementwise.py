from __future__ import annotations

from bakenn.errors import GraphValidationError
from bakenn.ir.graph import QuantizedGraph
from bakenn.ir.ops.elementwise import AddOp, ClampOp, MulOp, RequantizeOp
from bakenn.ir.types import DType, PerTensorQParams, TensorType
from bakenn.ir.verify import verify_op


def _fail(message: str) -> None:
    raise GraphValidationError(message)


def _activation(op_name: str, graph: QuantizedGraph, value: str) -> TensorType:
    tensor = graph.values[value]
    if tensor.dtype is not DType.INT8:
        _fail(f"{op_name}: elementwise activations must be int8")
    if not isinstance(tensor.qparams, PerTensorQParams):
        _fail(f"{op_name}: elementwise activations require per-tensor qparams")
    return tensor


def _verify_binary(op: AddOp | MulOp, graph: QuantizedGraph) -> None:
    input_a = _activation(op.name, graph, op.input_a)
    input_b = _activation(op.name, graph, op.input_b)
    output = _activation(op.name, graph, op.output)
    if len(input_a.shape) != len(input_b.shape) or len(input_a.shape) != len(output.shape):
        _fail(f"{op.name}: static broadcasting requires equal input and output ranks")
    expected = tuple(max(a, b) for a, b in zip(input_a.shape, input_b.shape))
    if any(a != b and a != 1 and b != 1 for a, b in zip(input_a.shape, input_b.shape)):
        _fail(f"{op.name}: input shapes are not statically broadcast-compatible")
    if output.shape != expected:
        _fail(f"{op.name}: output shape does not equal the static broadcast result {expected}")
    if input_a.layout is not input_b.layout or input_a.layout is not output.layout:
        _fail(f"{op.name}: input and output layouts must match exactly")
    if not -128 <= op.activation_min <= op.activation_max <= 127:
        _fail(f"{op.name}: invalid fused activation clamp")


@verify_op.register
def _verify_add(op: AddOp, graph: QuantizedGraph) -> None:
    _verify_binary(op, graph)


@verify_op.register
def _verify_mul(op: MulOp, graph: QuantizedGraph) -> None:
    _verify_binary(op, graph)


@verify_op.register
def _verify_clamp(op: ClampOp, graph: QuantizedGraph) -> None:
    input_type = _activation(op.name, graph, op.input)
    output_type = _activation(op.name, graph, op.output)
    if input_type.shape != output_type.shape or input_type.layout is not output_type.layout:
        _fail(f"{op.name}: Clamp input and output shape/layout must match exactly")
    if input_type.qparams != output_type.qparams:
        _fail(
            f"{op.name}: Clamp preserves qparams; insert Requantize before Clamp when qparams differ"
        )
    if not -128 <= op.activation_min <= op.activation_max <= 127:
        _fail(f"{op.name}: invalid clamp bounds")


@verify_op.register
def _verify_requantize(op: RequantizeOp, graph: QuantizedGraph) -> None:
    input_type = _activation(op.name, graph, op.input)
    output_type = _activation(op.name, graph, op.output)
    if input_type.shape != output_type.shape or input_type.layout is not output_type.layout:
        _fail(f"{op.name}: Requantize input and output shape/layout must match exactly")
    if not -128 <= op.activation_min <= op.activation_max <= 127:
        _fail(f"{op.name}: invalid fused Requantize clamp")


__all__: list[str] = []
