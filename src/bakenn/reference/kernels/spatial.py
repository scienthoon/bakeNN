from __future__ import annotations

from typing import Mapping

import numpy as np

from bakenn.ir.types import PerTensorQParams
from bakenn.plan.steps.spatial import ConvTranspose2DStep, ResizeBilinear2DStep, ResizeNearest2DStep
from bakenn.plan.types import ExecutionPlan
from bakenn.quantization.fixedpoint import INT32_MAX, INT32_MIN, multiply_by_quantized_multiplier, rounding_divide_by_pot
from bakenn.reference.executor import execute_step


@execute_step.register
def _nearest(step: ResizeNearest2DStep, plan: ExecutionPlan, values: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    source = np.asarray(values[step.input], dtype=np.int8)
    output_shape = plan.tensors[step.output].tensor_type.shape
    _, _, _, channels = source.shape
    _, output_h, output_w, _ = output_shape
    result = np.empty(output_shape, dtype=np.int8)
    for y in range(output_h):
        source_y = step.y_indices[y]
        for x in range(output_w):
            source_x = step.x_indices[x]
            result[0, y, x, :] = source[0, source_y, source_x, :channels]
    return {step.output: result}


@execute_step.register
def _bilinear(step: ResizeBilinear2DStep, plan: ExecutionPlan, values: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    source = np.asarray(values[step.input], dtype=np.int8)
    output_shape = plan.tensors[step.output].tensor_type.shape
    _, output_h, output_w, channels = output_shape
    result = np.empty(output_shape, dtype=np.int8)
    for y in range(output_h):
        wy = step.yw_q15[y]
        for x in range(output_w):
            wx = step.xw_q15[x]
            w00 = (32768 - wy) * (32768 - wx)
            w01 = (32768 - wy) * wx
            w10 = wy * (32768 - wx)
            w11 = wy * wx
            for channel in range(channels):
                numerator = (
                    int(source[0, step.y0[y], step.x0[x], channel]) * w00
                    + int(source[0, step.y0[y], step.x1[x], channel]) * w01
                    + int(source[0, step.y1[y], step.x0[x], channel]) * w10
                    + int(source[0, step.y1[y], step.x1[x], channel]) * w11
                )
                result[0, y, x, channel] = np.int8(max(-128, min(127, rounding_divide_by_pot(numerator, 30))))
    return {step.output: result}


@execute_step.register
def _transpose(step: ConvTranspose2DStep, plan: ExecutionPlan, values: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    source = np.asarray(values[step.input], dtype=np.int8)
    weight = np.asarray(values[step.weight], dtype=np.int8)
    bias = np.asarray(values[step.bias], dtype=np.int32)
    input_qparams = plan.tensors[step.input].tensor_type.qparams
    output_type = plan.tensors[step.output].tensor_type
    output_qparams = output_type.qparams
    assert isinstance(input_qparams, PerTensorQParams)
    assert isinstance(output_qparams, PerTensorQParams)
    _, input_h, input_w, input_channels = source.shape
    output_channels, kernel_h, kernel_w, input_channels_per_group = weight.shape
    output_channels_per_group = output_channels // step.groups
    _, output_h, output_w, _ = output_type.shape
    stride_h, stride_w = step.stride
    dilation_h, dilation_w = step.dilation
    pad_top, _, pad_left, _ = step.padding
    result = np.empty(output_type.shape, dtype=np.int8)
    for output_y in range(output_h):
        for output_x in range(output_w):
            for output_channel in range(output_channels):
                accumulator = int(bias[output_channel])
                group = output_channel // output_channels_per_group
                input_channel_start = group * input_channels_per_group
                for kernel_y in range(kernel_h):
                    candidate_y = output_y + pad_top - kernel_y * dilation_h
                    if candidate_y < 0 or candidate_y % stride_h:
                        continue
                    input_y = candidate_y // stride_h
                    if input_y >= input_h:
                        continue
                    for kernel_x in range(kernel_w):
                        candidate_x = output_x + pad_left - kernel_x * dilation_w
                        if candidate_x < 0 or candidate_x % stride_w:
                            continue
                        input_x = candidate_x // stride_w
                        if input_x >= input_w:
                            continue
                        for local_input_channel in range(input_channels_per_group):
                            input_channel = input_channel_start + local_input_channel
                            accumulator += (int(source[0, input_y, input_x, input_channel]) - input_qparams.zero_point) * int(
                                weight[output_channel, kernel_y, kernel_x, local_input_channel]
                            )
                if not INT32_MIN <= accumulator <= INT32_MAX:
                    raise AssertionError("ConvTranspose2D accumulator proof was violated")
                scaled = multiply_by_quantized_multiplier(accumulator, step.multipliers[output_channel], step.shifts[output_channel])
                result[0, output_y, output_x, output_channel] = np.int8(
                    min(step.activation_max, max(step.activation_min, scaled + output_qparams.zero_point))
                )
    return {step.output: result}


__all__: list[str] = []
