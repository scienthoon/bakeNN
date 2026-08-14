from __future__ import annotations

from functools import singledispatch
from typing import Mapping

import numpy as np

from bakenn.errors import CompileError
from bakenn.ir import PerTensorQParams
from bakenn.plan import ExecutionPlan, LinearStep
from bakenn.quantization.fixedpoint import (
    INT32_MAX,
    INT32_MIN,
    multiply_by_quantized_multiplier,
)


def quantize_input(plan: ExecutionPlan, values: np.ndarray) -> np.ndarray:
    tensor = plan.tensors[plan.inputs[0]].tensor_type
    assert isinstance(tensor.qparams, PerTensorQParams)
    # The public generated header exposes float32 scales for MCU firmware, so
    # host-side input conversion intentionally uses the same float32 division
    # and rounding domain.  Promoting only this side to binary64 can change a
    # half-LSB decision even when the stored scale bits are identical.
    try:
        source = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as error:
        raise CompileError("input must be a real numeric array") from error
    if source.dtype.kind not in "iuf":
        raise CompileError("input must use a real numeric dtype")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.array(source, dtype=np.float32, copy=True, order="C")
    if array.shape != tensor.shape:
        raise CompileError(f"input has shape {array.shape}, expected {tensor.shape}")
    if not np.all(np.isfinite(array)):
        raise CompileError("input contains NaN or infinity")
    centered = array / np.float32(tensor.qparams.scale)
    half = np.float32(0.5)
    rounded = np.where(centered >= 0.0, np.floor(centered + half), np.ceil(centered - half))
    rounded += tensor.qparams.zero_point
    return np.clip(rounded, -128, 127).astype(np.int8)


def dequantize_output(plan: ExecutionPlan, values: np.ndarray) -> np.ndarray:
    tensor = plan.tensors[plan.outputs[0]].tensor_type
    assert isinstance(tensor.qparams, PerTensorQParams)
    array = np.asarray(values)
    if array.dtype != np.int8:
        raise CompileError("dequantization input must have dtype int8")
    if array.shape != tensor.shape:
        raise CompileError(f"output has shape {array.shape}, expected {tensor.shape}")
    array = array.astype(np.int32)
    return (array - tensor.qparams.zero_point).astype(np.float32) * np.float32(
        tensor.qparams.scale
    )


@singledispatch
def execute_step(
    step: object,
    plan: ExecutionPlan,
    values: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    del plan, values
    raise CompileError(f"no reference executor registered for step type {type(step).__name__}")


@execute_step.register
def _execute_linear(
    step: LinearStep,
    plan: ExecutionPlan,
    values: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    x = values[step.input].reshape(-1)
    weight = values[step.weight]
    bias = values[step.bias]
    input_qparams = plan.tensors[step.input].tensor_type.qparams
    output_qparams = plan.tensors[step.output].tensor_type.qparams
    assert isinstance(input_qparams, PerTensorQParams)
    assert isinstance(output_qparams, PerTensorQParams)
    result = np.empty(weight.shape[0], dtype=np.int8)
    for channel in range(weight.shape[0]):
        accumulator = int(bias[channel])
        accumulator += sum(
            (int(x_value) - input_qparams.zero_point) * int(w_value)
            for x_value, w_value in zip(x, weight[channel])
        )
        if not INT32_MIN <= accumulator <= INT32_MAX:
            raise AssertionError("compile-time accumulator proof was violated")
        requantized = multiply_by_quantized_multiplier(
            accumulator, step.multipliers[channel], step.shifts[channel]
        )
        shifted = requantized + output_qparams.zero_point
        clamped = min(step.activation_max, max(step.activation_min, shifted))
        result[channel] = clamped
    return {step.output: result.reshape(plan.tensors[step.output].tensor_type.shape)}


def run_reference(plan: ExecutionPlan, input_values: np.ndarray) -> np.ndarray:
    """Execute a lowered plan with generic per-step integer dispatch."""

    input_name = plan.inputs[0]
    expected_shape = plan.tensors[input_name].tensor_type.shape
    input_array = np.asarray(input_values)
    if input_array.dtype != np.int8:
        raise CompileError("integer reference input must have dtype int8")
    if input_array.shape != expected_shape:
        raise CompileError(f"integer reference input has shape {input_array.shape}, expected {expected_shape}")

    values: dict[str, np.ndarray] = {input_name: input_array}
    values.update(plan.constants)
    for step in plan.steps:
        produced = dict(execute_step(step, plan, values))
        expected_outputs = set(step.outputs)
        if set(produced) != expected_outputs:
            raise CompileError(
                f"reference executor for {step.name} produced {sorted(produced)}, "
                f"expected {sorted(expected_outputs)}"
            )
        for name, array in produced.items():
            tensor_type = plan.tensors[name].tensor_type
            result = np.asarray(array)
            expected_dtype = np.int8 if tensor_type.dtype.value == "int8" else np.int32
            if result.dtype != expected_dtype or result.shape != tensor_type.shape:
                raise CompileError(
                    f"reference executor for {step.name} produced invalid tensor {name}"
                )
            values[name] = result
    return np.array(values[plan.outputs[0]], copy=True)


# Install built-in step executors after the generic dispatch hook is defined.
from . import kernels as _built_in_kernels  # noqa: E402,F401


__all__ = ["dequantize_output", "execute_step", "quantize_input", "run_reference"]
