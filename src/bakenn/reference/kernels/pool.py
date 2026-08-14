from __future__ import annotations

from typing import Mapping

import numpy as np

from bakenn.ir.types import PerTensorQParams
from bakenn.plan.types import ExecutionPlan
from bakenn.plan.steps.pool import AveragePool2DStep, MaxPool2DStep
from bakenn.reference.executor import execute_step


def _round_divide_half_away(value: int, divisor: int) -> int:
    magnitude = (abs(value) + divisor // 2) // divisor
    return -magnitude if value < 0 else magnitude


def _window(
    input_values: np.ndarray,
    output_y: int,
    output_x: int,
    kernel: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int, int, int],
) -> np.ndarray:
    _, input_h, input_w, _ = input_values.shape
    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    pad_top, _, pad_left, _ = padding
    start_y = output_y * stride_h - pad_top
    start_x = output_x * stride_w - pad_left
    y0, y1 = max(start_y, 0), min(start_y + kernel_h, input_h)
    x0, x1 = max(start_x, 0), min(start_x + kernel_w, input_w)
    return input_values[0, y0:y1, x0:x1, :]


@execute_step.register
def _execute_average_pool(
    step: AveragePool2DStep,
    plan: ExecutionPlan,
    values: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    input_values = values[step.input]
    output_type = plan.tensors[step.output].tensor_type
    input_qparams = plan.tensors[step.input].tensor_type.qparams
    assert isinstance(input_qparams, PerTensorQParams)
    output = np.empty(output_type.shape, dtype=np.int8)
    _, output_h, output_w, channels = output_type.shape
    for output_y in range(output_h):
        for output_x in range(output_w):
            window = _window(
                input_values,
                output_y,
                output_x,
                step.kernel,
                step.stride,
                step.padding,
            )
            valid_count = window.shape[0] * window.shape[1]
            if valid_count == 0:
                raise AssertionError("compile-time non-empty pool-window proof was violated")
            for channel in range(channels):
                accumulator = sum(
                    int(value) - input_qparams.zero_point
                    for value in window[:, :, channel].reshape(-1)
                )
                if abs(accumulator) > step.accumulator_bound:
                    raise AssertionError("compile-time average accumulator proof was violated")
                result = _round_divide_half_away(accumulator, valid_count)
                result += input_qparams.zero_point
                result = min(step.activation_max, max(step.activation_min, result))
                output[0, output_y, output_x, channel] = result
    return {step.output: output}


@execute_step.register
def _execute_max_pool(
    step: MaxPool2DStep,
    plan: ExecutionPlan,
    values: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    input_values = values[step.input]
    output_type = plan.tensors[step.output].tensor_type
    output = np.empty(output_type.shape, dtype=np.int8)
    _, output_h, output_w, channels = output_type.shape
    for output_y in range(output_h):
        for output_x in range(output_w):
            window = _window(
                input_values,
                output_y,
                output_x,
                step.kernel,
                step.stride,
                step.padding,
            )
            if window.shape[0] == 0 or window.shape[1] == 0:
                raise AssertionError("compile-time non-empty pool-window proof was violated")
            for channel in range(channels):
                result = max(int(value) for value in window[:, :, channel].reshape(-1))
                result = min(step.activation_max, max(step.activation_min, result))
                output[0, output_y, output_x, channel] = result
    return {step.output: output}


__all__ = ["_round_divide_half_away"]
