from __future__ import annotations

from typing import Mapping

import numpy as np

from bakenn.ir.types import PerTensorQParams
from bakenn.plan.steps.sequence import AveragePool1DStep, Conv1DStep, MaxPool1DStep
from bakenn.plan.types import ExecutionPlan
from bakenn.quantization.fixedpoint import multiply_by_quantized_multiplier
from bakenn.reference.executor import execute_step
from bakenn.reference.kernels.pool import _round_divide_half_away


@execute_step.register
def _execute_conv1d(step: Conv1DStep, plan: ExecutionPlan, values: Mapping[str, np.ndarray]):  # type: ignore[no-untyped-def]
    x, weight, bias = values[step.input], values[step.weight], values[step.bias]
    input_type = plan.tensors[step.input].tensor_type
    output_type = plan.tensors[step.output].tensor_type
    input_qparams, output_qparams = input_type.qparams, output_type.qparams
    assert isinstance(input_qparams, PerTensorQParams)
    assert isinstance(output_qparams, PerTensorQParams)
    _, input_length, input_channels = x.shape
    output_channels, kernel, group_input_channels = weight.shape
    output_channels_per_group = output_channels // step.groups
    output = np.empty(output_type.shape, dtype=np.int8)
    pad_left, _ = step.padding
    for position in range(output_type.shape[1]):
        for output_channel in range(output_channels):
            group = output_channel // output_channels_per_group
            input_base = group * group_input_channels
            accumulator = int(bias[output_channel])
            for kernel_index in range(kernel):
                input_position = position * step.stride + kernel_index * step.dilation - pad_left
                for local_channel in range(group_input_channels):
                    code = input_qparams.zero_point
                    if 0 <= input_position < input_length:
                        code = int(x[0, input_position, input_base + local_channel])
                    accumulator += (code - input_qparams.zero_point) * int(weight[output_channel, kernel_index, local_channel])
            scaled = multiply_by_quantized_multiplier(accumulator, step.multipliers[output_channel], step.shifts[output_channel])
            output[0, position, output_channel] = np.int8(min(step.activation_max, max(step.activation_min, scaled + output_qparams.zero_point)))
    return {step.output: output}


def _window(x: np.ndarray, position: int, kernel: int, stride: int, pad_left: int) -> np.ndarray:
    start = position * stride - pad_left
    return x[0, max(0, start):min(x.shape[1], start + kernel), :]


@execute_step.register
def _execute_average_pool1d(step: AveragePool1DStep, plan: ExecutionPlan, values: Mapping[str, np.ndarray]):  # type: ignore[no-untyped-def]
    x = values[step.input]
    output_type = plan.tensors[step.output].tensor_type
    qparams = plan.tensors[step.input].tensor_type.qparams
    assert isinstance(qparams, PerTensorQParams)
    output = np.empty(output_type.shape, dtype=np.int8)
    for position in range(output_type.shape[1]):
        window = _window(x, position, step.kernel, step.stride, step.padding[0])
        for channel in range(output_type.shape[2]):
            total = sum(int(code) - qparams.zero_point for code in window[:, channel])
            result = _round_divide_half_away(total, window.shape[0]) + qparams.zero_point
            output[0, position, channel] = np.int8(min(step.activation_max, max(step.activation_min, result)))
    return {step.output: output}


@execute_step.register
def _execute_max_pool1d(step: MaxPool1DStep, plan: ExecutionPlan, values: Mapping[str, np.ndarray]):  # type: ignore[no-untyped-def]
    x = values[step.input]
    output_type = plan.tensors[step.output].tensor_type
    output = np.empty(output_type.shape, dtype=np.int8)
    for position in range(output_type.shape[1]):
        window = _window(x, position, step.kernel, step.stride, step.padding[0])
        for channel in range(output_type.shape[2]):
            result = max(int(code) for code in window[:, channel])
            output[0, position, channel] = np.int8(min(step.activation_max, max(step.activation_min, result)))
    return {step.output: output}


__all__: list[str] = []
