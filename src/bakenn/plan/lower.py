from __future__ import annotations

from functools import singledispatch

import numpy as np

from bakenn.errors import CompileError
from bakenn.ir import LinearOp, PerAxisQParams, PerTensorQParams, QuantizedGraph, verify_graph
from bakenn.quantization.fixedpoint import INT32_MAX, quantize_multiplier
from .memory import plan_memory
from .types import ExecutionPlan, ExecutionStep, LinearStep


def _accumulator_bounds(input_zero_point: int, weight: np.ndarray, bias: np.ndarray) -> tuple[int, ...]:
    max_abs_input = max(abs(-128 - input_zero_point), abs(127 - input_zero_point))
    bounds: list[int] = []
    for channel in range(weight.shape[0]):
        bound = abs(int(bias[channel])) + max_abs_input * sum(abs(int(value)) for value in weight[channel])
        if bound > INT32_MAX:
            raise CompileError(
                f"channel {channel} accumulator bound {bound} exceeds int32; model cannot be compiled safely"
            )
        bounds.append(bound)
    return tuple(bounds)


@singledispatch
def lower_op(op: object, graph: QuantizedGraph) -> ExecutionStep:
    del graph
    raise CompileError(f"no plan lowering registered for operation type {type(op).__name__}")


@lower_op.register
def _lower_linear(op: LinearOp, graph: QuantizedGraph) -> LinearStep:
    input_type = graph.values[op.input]
    weight_type = graph.values[op.weight]
    output_type = graph.values[op.output]
    assert isinstance(input_type.qparams, PerTensorQParams)
    assert isinstance(weight_type.qparams, PerAxisQParams)
    assert isinstance(output_type.qparams, PerTensorQParams)
    weight = graph.constants[op.weight]
    bias = graph.constants[op.bias]
    bounds = _accumulator_bounds(input_type.qparams.zero_point, weight, bias)
    multipliers: list[int] = []
    shifts: list[int] = []
    for channel, weight_scale in enumerate(weight_type.qparams.scales):
        real_multiplier = input_type.qparams.scale * weight_scale / output_type.qparams.scale
        multiplier, shift = quantize_multiplier(real_multiplier)
        if shift > 0 and bounds[channel] * (1 << shift) > INT32_MAX:
            raise CompileError(
                f"{op.name} channel {channel}: requantization left shift is not int32-safe"
            )
        multipliers.append(multiplier)
        shifts.append(shift)
    return LinearStep(
        name=op.name,
        input=op.input,
        weight=op.weight,
        bias=op.bias,
        output=op.output,
        multipliers=tuple(multipliers),
        shifts=tuple(shifts),
        activation_min=op.activation_min,
        activation_max=op.activation_max,
        accumulator_bounds=bounds,
    )


def lower_to_plan(graph: QuantizedGraph, *, arena_alignment: int = 16) -> ExecutionPlan:
    verify_graph(graph)
    steps = tuple(lower_op(op, graph) for op in graph.ops)
    layout = plan_memory(
        values=graph.values,
        constants=set(graph.constants),
        inputs=graph.inputs,
        outputs=graph.outputs,
        steps=steps,
        arena_alignment=arena_alignment,
    )
    return ExecutionPlan(
        name=graph.name,
        tensors=layout.tensors,
        constants=graph.constants,
        steps=steps,
        inputs=graph.inputs,
        outputs=graph.outputs,
        arena_size=layout.arena_size,
        arena_alignment=layout.arena_alignment,
        arithmetic_profile=graph.arithmetic_profile,
        activation_arena_size=layout.activation_arena_size,
        scratch_size=layout.scratch_size,
        scratch_offset=layout.scratch_offset,
        scratch_alignment=layout.scratch_alignment,
        alias_groups=layout.alias_groups,
    )


# Install built-in lowerings after the generic dispatch hook is defined.
from . import lowering as _built_in_lowerings  # noqa: E402,F401


__all__ = ["lower_op", "lower_to_plan"]
