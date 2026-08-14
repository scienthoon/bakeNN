from __future__ import annotations

from typing import Mapping

import numpy as np

from bakenn.ir.types import PerTensorQParams
from bakenn.plan.steps.elementwise import AddStep, ClampStep, MulStep, RequantizeStep
from bakenn.plan.types import ExecutionPlan
from bakenn.quantization.fixedpoint import INT32_MAX, INT32_MIN, multiply_by_quantized_multiplier
from bakenn.reference.executor import execute_step


def _qparams(plan: ExecutionPlan, value: str) -> PerTensorQParams:
    qparams = plan.tensors[value].tensor_type.qparams
    assert isinstance(qparams, PerTensorQParams)
    return qparams


def _shape(plan: ExecutionPlan, value: str) -> tuple[int, ...]:
    return plan.tensors[value].tensor_type.shape


def _saturate(value: int, minimum: int = -128, maximum: int = 127) -> int:
    return min(maximum, max(minimum, value))


@execute_step.register
def _execute_add(
    step: AddStep,
    plan: ExecutionPlan,
    values: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    output_shape = _shape(plan, step.output)
    input_a = np.broadcast_to(
        np.asarray(values[step.input_a], dtype=np.int8), output_shape
    ).reshape(-1)
    input_b = np.broadcast_to(
        np.asarray(values[step.input_b], dtype=np.int8), output_shape
    ).reshape(-1)
    input_a_qparams = _qparams(plan, step.input_a)
    input_b_qparams = _qparams(plan, step.input_b)
    output_qparams = _qparams(plan, step.output)
    result = np.empty(input_a.size, dtype=np.int8)
    for index, (code_a, code_b) in enumerate(zip(input_a, input_b)):
        centered_a = int(code_a) - input_a_qparams.zero_point
        centered_b = int(code_b) - input_b_qparams.zero_point
        shifted_a = centered_a * (1 << step.left_shift)
        shifted_b = centered_b * (1 << step.left_shift)
        scaled_a = multiply_by_quantized_multiplier(
            shifted_a, step.input_a_multiplier, step.input_a_shift
        )
        scaled_b = multiply_by_quantized_multiplier(
            shifted_b, step.input_b_multiplier, step.input_b_shift
        )
        summed = scaled_a + scaled_b
        if not INT32_MIN <= summed <= INT32_MAX:
            raise AssertionError("compile-time Add intermediate proof was violated")
        requantized = multiply_by_quantized_multiplier(
            summed, step.output_multiplier, step.output_shift
        )
        shifted = requantized + output_qparams.zero_point
        result[index] = _saturate(shifted, step.activation_min, step.activation_max)
    return {step.output: result.reshape(_shape(plan, step.output))}


@execute_step.register
def _execute_mul(
    step: MulStep,
    plan: ExecutionPlan,
    values: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    output_shape = _shape(plan, step.output)
    input_a = np.broadcast_to(
        np.asarray(values[step.input_a], dtype=np.int8), output_shape
    ).reshape(-1)
    input_b = np.broadcast_to(
        np.asarray(values[step.input_b], dtype=np.int8), output_shape
    ).reshape(-1)
    input_a_qparams = _qparams(plan, step.input_a)
    input_b_qparams = _qparams(plan, step.input_b)
    output_qparams = _qparams(plan, step.output)
    result = np.empty(input_a.size, dtype=np.int8)
    for index, (code_a, code_b) in enumerate(zip(input_a, input_b)):
        centered_a = int(code_a) - input_a_qparams.zero_point
        centered_b = int(code_b) - input_b_qparams.zero_point
        product = centered_a * centered_b
        if not INT32_MIN <= product <= INT32_MAX:
            raise AssertionError("compile-time Mul product proof was violated")
        requantized = multiply_by_quantized_multiplier(
            product, step.output_multiplier, step.output_shift
        )
        shifted = requantized + output_qparams.zero_point
        result[index] = _saturate(shifted, step.activation_min, step.activation_max)
    return {step.output: result.reshape(_shape(plan, step.output))}


@execute_step.register
def _execute_clamp(
    step: ClampStep,
    plan: ExecutionPlan,
    values: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    result = np.clip(
        np.asarray(values[step.input], dtype=np.int8),
        step.activation_min,
        step.activation_max,
    ).astype(np.int8, copy=True)
    return {step.output: result.reshape(_shape(plan, step.output))}


@execute_step.register
def _execute_requantize(
    step: RequantizeStep,
    plan: ExecutionPlan,
    values: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    input_values = np.asarray(values[step.input], dtype=np.int8).reshape(-1)
    input_qparams = _qparams(plan, step.input)
    output_qparams = _qparams(plan, step.output)
    result = np.empty(input_values.size, dtype=np.int8)
    for index, code in enumerate(input_values):
        centered = int(code) - input_qparams.zero_point
        requantized = multiply_by_quantized_multiplier(centered, step.multiplier, step.shift)
        result[index] = _saturate(
            requantized + output_qparams.zero_point,
            step.activation_min,
            step.activation_max,
        )
    return {step.output: result.reshape(_shape(plan, step.output))}


__all__: list[str] = []
