from __future__ import annotations

from dataclasses import is_dataclass
from functools import singledispatch
import math

import numpy as np

from bakenn.errors import GraphValidationError
from .graph import QuantizedGraph
from .op import LinearOp
from .types import (
    DType,
    Layout,
    PerAxisQParams,
    PerTensorQParams,
    normalize_scale_float32,
)


def _fail(message: str) -> None:
    raise GraphValidationError(message)


@singledispatch
def verify_op(op: object, graph: QuantizedGraph) -> None:
    """Dispatch per-op validation without a stringly typed central switch."""

    del graph
    _fail(f"unsupported operation type: {type(op).__name__}")


@verify_op.register
def _verify_linear(op: LinearOp, graph: QuantizedGraph) -> None:
    x = graph.values[op.input]
    weight = graph.values[op.weight]
    bias = graph.values[op.bias]
    output = graph.values[op.output]

    if op.weight not in graph.constants or op.bias not in graph.constants:
        _fail(f"{op.name}: Linear weight and bias must be compile-time constants")
    if x.dtype is not DType.INT8 or output.dtype is not DType.INT8:
        _fail(f"{op.name}: Linear activations must be int8")
    if x.layout is not Layout.NC or output.layout is not Layout.NC:
        _fail(f"{op.name}: Linear activations must use NC layout")
    if len(x.shape) != 2 or len(output.shape) != 2 or x.shape[0] != 1 or output.shape[0] != 1:
        _fail(f"{op.name}: v0 Linear requires static batch size one")
    if not isinstance(x.qparams, PerTensorQParams) or not isinstance(output.qparams, PerTensorQParams):
        _fail(f"{op.name}: activations require per-tensor qparams")

    if weight.dtype is not DType.INT8 or weight.layout is not Layout.OI or len(weight.shape) != 2:
        _fail(f"{op.name}: weight must be rank-two OI int8")
    if not isinstance(weight.qparams, PerAxisQParams):
        _fail(f"{op.name}: weight requires per-axis qparams")
    if weight.qparams.axis != 0 or len(weight.qparams.scales) != weight.shape[0]:
        _fail(f"{op.name}: weight qparams must be per-output-channel on axis zero")
    if any(zero_point != 0 for zero_point in weight.qparams.zero_points):
        _fail(f"{op.name}: symmetric weights require zero_point zero")
    if np.any(graph.constants[op.weight] == -128):
        _fail(f"{op.name}: symmetric weights must stay in [-127, 127]")

    if bias.dtype is not DType.INT32 or bias.layout is not Layout.C or bias.shape != (weight.shape[0],):
        _fail(f"{op.name}: bias must be one int32 value per output channel")
    if not isinstance(bias.qparams, PerAxisQParams) or bias.qparams.axis != 0:
        _fail(f"{op.name}: bias requires per-channel qparams")
    if any(zero_point != 0 for zero_point in bias.qparams.zero_points):
        _fail(f"{op.name}: bias zero_points must be zero")

    if weight.shape[1] != x.shape[1] or output.shape[1] != weight.shape[0]:
        _fail(f"{op.name}: incompatible Linear shapes")
    try:
        expected_bias_scales = tuple(
            normalize_scale_float32(x.qparams.scale * scale)
            for scale in weight.qparams.scales
        )
    except ValueError:
        _fail(f"{op.name}: bias scale product is not representable as float32")
    if len(bias.qparams.scales) != len(expected_bias_scales) or any(
        not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.0)
        for actual, expected in zip(bias.qparams.scales, expected_bias_scales)
    ):
        _fail(f"{op.name}: bias scale must equal input_scale * weight_scale[channel]")
    if not -128 <= op.activation_min <= op.activation_max <= 127:
        _fail(f"{op.name}: invalid fused activation clamp")


def _op_fields(op: object) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    if not is_dataclass(op) or not getattr(type(op), "__dataclass_params__", None).frozen:
        _fail(f"operation {type(op).__name__} must be an immutable dataclass")
    name = getattr(op, "name", None)
    inputs = getattr(op, "inputs", None)
    outputs = getattr(op, "outputs", None)
    if not isinstance(name, str) or not name:
        _fail("operation names must be non-empty strings")
    if not isinstance(inputs, tuple) or not inputs or not all(isinstance(item, str) and item for item in inputs):
        _fail(f"{name}: inputs must be a non-empty tuple of value names")
    if not isinstance(outputs, tuple) or not outputs or not all(isinstance(item, str) and item for item in outputs):
        _fail(f"{name}: outputs must be a non-empty tuple of value names")
    if len(set(outputs)) != len(outputs):
        _fail(f"{name}: operation outputs must be unique")
    return name, inputs, outputs


def verify_graph(graph: QuantizedGraph) -> None:
    if graph.arithmetic_profile != "bakenn.int8.v1":
        _fail(f"unsupported arithmetic profile: {graph.arithmetic_profile}")
    if len(graph.inputs) != 1 or len(graph.outputs) != 1:
        _fail("P0 supports exactly one graph input and one graph output")
    if not graph.ops:
        _fail("graph must contain at least one operation")
    if len(set(graph.inputs)) != len(graph.inputs) or len(set(graph.outputs)) != len(graph.outputs):
        _fail("graph input and output names must be unique")

    for name in (*graph.inputs, *graph.outputs):
        if name not in graph.values:
            _fail(f"unknown graph value: {name}")

    constant_names = set(graph.constants)
    value_names = set(graph.values)
    if constant_names - value_names:
        _fail(f"constants missing types: {sorted(constant_names - value_names)}")
    if constant_names & set(graph.inputs):
        _fail("graph inputs cannot also be constants")

    for name, array in graph.constants.items():
        tensor_type = graph.values[name]
        expected_dtype = np.int8 if tensor_type.dtype is DType.INT8 else np.int32
        if array.shape != tensor_type.shape:
            _fail(f"constant {name} has shape {array.shape}, expected {tensor_type.shape}")
        if array.dtype != expected_dtype:
            _fail(f"constant {name} has dtype {array.dtype}, expected {expected_dtype}")

    defined = set(graph.inputs) | constant_names
    producer_by_value: dict[str, object] = {}
    op_names: set[str] = set()
    normalized: list[tuple[object, str, tuple[str, ...], tuple[str, ...]]] = []

    for op in graph.ops:
        name, inputs, outputs = _op_fields(op)
        if name in op_names:
            _fail(f"duplicate operation name: {name}")
        op_names.add(name)
        for value in inputs:
            if value not in graph.values:
                _fail(f"{name} uses unknown value: {value}")
            if value not in defined:
                _fail(f"{name} uses {value} before it is defined (graph is cyclic or not topological)")
        for value in outputs:
            if value not in graph.values:
                _fail(f"missing output type for {value}")
            if value in defined or value in producer_by_value:
                _fail(f"SSA value has multiple definitions: {value}")
        verify_op(op, graph)
        for value in outputs:
            producer_by_value[value] = op
            defined.add(value)
        normalized.append((op, name, inputs, outputs))

    for name in graph.values:
        if name not in defined:
            _fail(f"value has no producer: {name}")
    for name in graph.outputs:
        if name not in producer_by_value:
            _fail(f"graph output must be produced by an operation: {name}")

    needed_values = set(graph.outputs)
    needed_op_ids: set[int] = set()
    pending = list(graph.outputs)
    while pending:
        value = pending.pop()
        producer = producer_by_value.get(value)
        if producer is None or id(producer) in needed_op_ids:
            continue
        needed_op_ids.add(id(producer))
        _, inputs, _ = _op_fields(producer)
        for operand in inputs:
            if operand not in needed_values:
                needed_values.add(operand)
                pending.append(operand)
    if len(needed_op_ids) != len(graph.ops):
        dead = sorted(name for op, name, _, _ in normalized if id(op) not in needed_op_ids)
        _fail(f"graph contains dead or disconnected operations: {dead}")
    unused_inputs = sorted(name for name in graph.inputs if name not in needed_values)
    if unused_inputs:
        _fail(f"graph inputs do not contribute to outputs: {unused_inputs}")
    unused_constants = sorted(name for name in constant_names if name not in needed_values)
    if unused_constants:
        _fail(f"graph contains unused constants: {unused_constants}")


# Install built-in family registrations only after ``verify_op`` exists.  This
# keeps family ownership modular while making the public verifier complete.
from . import verifiers as _built_in_verifiers  # noqa: E402,F401


__all__ = ["verify_graph", "verify_op"]
