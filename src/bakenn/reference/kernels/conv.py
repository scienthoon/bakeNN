from __future__ import annotations

from typing import Mapping

import numpy as np

from bakenn.ir.types import PerTensorQParams
from bakenn.plan.steps.conv import Conv2DStep, DepthwiseConv2DStep
from bakenn.plan.types import ExecutionPlan
from bakenn.quantization.fixedpoint import INT32_MAX, INT32_MIN, multiply_by_quantized_multiplier
from bakenn.reference.executor import execute_step


def _finish(
    accumulator: int,
    multiplier: int,
    shift: int,
    output_zero_point: int,
    activation_min: int,
    activation_max: int,
) -> np.int8:
    if not INT32_MIN <= accumulator <= INT32_MAX:
        raise AssertionError("compile-time convolution accumulator proof was violated")
    requantized = multiply_by_quantized_multiplier(accumulator, multiplier, shift)
    return np.int8(min(activation_max, max(activation_min, requantized + output_zero_point)))


@execute_step.register
def _execute_conv2d(
    step: Conv2DStep,
    plan: ExecutionPlan,
    values: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    x = values[step.input]
    weight = values[step.weight]
    bias = values[step.bias]
    output_type = plan.tensors[step.output].tensor_type
    input_qparams = plan.tensors[step.input].tensor_type.qparams
    output_qparams = output_type.qparams
    assert isinstance(input_qparams, PerTensorQParams)
    assert isinstance(output_qparams, PerTensorQParams)
    _, input_height, input_width, input_channels = x.shape
    output_channels, kernel_height, kernel_width, group_input_channels = weight.shape
    output_channels_per_group = output_channels // step.groups
    _, output_height, output_width, _ = output_type.shape
    stride_height, stride_width = step.stride
    dilation_height, dilation_width = step.dilation
    pad_top, _, pad_left, _ = step.padding
    result = np.empty(output_type.shape, dtype=np.int8)

    for output_y in range(output_height):
        for output_x in range(output_width):
            for output_channel in range(output_channels):
                group = output_channel // output_channels_per_group
                input_channel_base = group * group_input_channels
                accumulator = int(bias[output_channel])
                for kernel_y in range(kernel_height):
                    input_y = output_y * stride_height + kernel_y * dilation_height - pad_top
                    for kernel_x in range(kernel_width):
                        input_x = output_x * stride_width + kernel_x * dilation_width - pad_left
                        for local_input_channel in range(group_input_channels):
                            input_channel = input_channel_base + local_input_channel
                            input_code = input_qparams.zero_point
                            if 0 <= input_y < input_height and 0 <= input_x < input_width:
                                input_code = int(x[0, input_y, input_x, input_channel])
                            accumulator += (input_code - input_qparams.zero_point) * int(
                                weight[
                                    output_channel,
                                    kernel_y,
                                    kernel_x,
                                    local_input_channel,
                                ]
                            )
                result[0, output_y, output_x, output_channel] = _finish(
                    accumulator,
                    step.multipliers[output_channel],
                    step.shifts[output_channel],
                    output_qparams.zero_point,
                    step.activation_min,
                    step.activation_max,
                )
    return {step.output: result}


@execute_step.register
def _execute_depthwise_conv2d(
    step: DepthwiseConv2DStep,
    plan: ExecutionPlan,
    values: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    x = values[step.input]
    weight = values[step.weight]
    bias = values[step.bias]
    output_type = plan.tensors[step.output].tensor_type
    input_qparams = plan.tensors[step.input].tensor_type.qparams
    output_qparams = output_type.qparams
    assert isinstance(input_qparams, PerTensorQParams)
    assert isinstance(output_qparams, PerTensorQParams)
    _, input_height, input_width, _ = x.shape
    kernel_height, kernel_width, output_channels = weight.shape
    _, output_height, output_width, _ = output_type.shape
    stride_height, stride_width = step.stride
    dilation_height, dilation_width = step.dilation
    pad_top, _, pad_left, _ = step.padding
    result = np.empty(output_type.shape, dtype=np.int8)

    for output_y in range(output_height):
        for output_x in range(output_width):
            for output_channel in range(output_channels):
                input_channel = output_channel // step.depth_multiplier
                accumulator = int(bias[output_channel])
                for kernel_y in range(kernel_height):
                    input_y = output_y * stride_height + kernel_y * dilation_height - pad_top
                    for kernel_x in range(kernel_width):
                        input_x = output_x * stride_width + kernel_x * dilation_width - pad_left
                        input_code = input_qparams.zero_point
                        if 0 <= input_y < input_height and 0 <= input_x < input_width:
                            input_code = int(x[0, input_y, input_x, input_channel])
                        accumulator += (input_code - input_qparams.zero_point) * int(
                            weight[kernel_y, kernel_x, output_channel]
                        )
                result[0, output_y, output_x, output_channel] = _finish(
                    accumulator,
                    step.multipliers[output_channel],
                    step.shifts[output_channel],
                    output_qparams.zero_point,
                    step.activation_min,
                    step.activation_max,
                )
    return {step.output: result}


__all__: list[str] = []
