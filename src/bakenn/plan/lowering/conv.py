from __future__ import annotations

import numpy as np

from bakenn.errors import CompileError
from bakenn.ir.graph import QuantizedGraph
from bakenn.ir.ops.conv import Conv2DOp, DepthwiseConv2DOp
from bakenn.ir.types import PerAxisQParams, PerTensorQParams
from bakenn.plan.lower import lower_op
from bakenn.plan.steps.conv import Conv2DStep, DepthwiseConv2DStep
from bakenn.quantization.fixedpoint import INT32_MAX, quantize_multiplier


def _channel_values(weight: np.ndarray, channel: int, channel_axis: int) -> np.ndarray:
    if channel_axis == 0:
        return weight[channel]
    if channel_axis == 2:
        return weight[:, :, channel]
    raise AssertionError("unsupported convolution channel axis")


def _accumulator_bounds(
    input_zero_point: int,
    weight: np.ndarray,
    bias: np.ndarray,
    output_channels: int,
    channel_axis: int,
    op_name: str,
) -> tuple[int, ...]:
    max_abs_input = max(abs(-128 - input_zero_point), abs(127 - input_zero_point))
    bounds: list[int] = []
    for channel in range(output_channels):
        channel_weight = _channel_values(weight, channel, channel_axis)
        bound = abs(int(bias[channel])) + max_abs_input * sum(
            abs(int(value)) for value in channel_weight.flat
        )
        if bound > INT32_MAX:
            raise CompileError(
                f"{op_name} channel {channel}: accumulator bound {bound} exceeds int32; "
                "model cannot be compiled safely"
            )
        bounds.append(bound)
    return tuple(bounds)


def _requantization(
    op_name: str,
    input_scale: float,
    weight_scales: tuple[float, ...],
    output_scale: float,
    bounds: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    multipliers: list[int] = []
    shifts: list[int] = []
    for channel, weight_scale in enumerate(weight_scales):
        multiplier, shift = quantize_multiplier(input_scale * weight_scale / output_scale)
        if shift > 0 and bounds[channel] * (1 << shift) > INT32_MAX:
            raise CompileError(
                f"{op_name} channel {channel}: requantization left shift is not int32-safe"
            )
        multipliers.append(multiplier)
        shifts.append(shift)
    return tuple(multipliers), tuple(shifts)


@lower_op.register
def _lower_conv2d(op: Conv2DOp, graph: QuantizedGraph) -> Conv2DStep:
    input_type = graph.values[op.input]
    weight_type = graph.values[op.weight]
    output_type = graph.values[op.output]
    assert isinstance(input_type.qparams, PerTensorQParams)
    assert isinstance(weight_type.qparams, PerAxisQParams)
    assert isinstance(output_type.qparams, PerTensorQParams)
    weight = graph.constants[op.weight]
    bias = graph.constants[op.bias]
    output_channels = weight.shape[0]
    bounds = _accumulator_bounds(
        input_type.qparams.zero_point,
        weight,
        bias,
        output_channels,
        0,
        op.name,
    )
    multipliers, shifts = _requantization(
        op.name,
        input_type.qparams.scale,
        weight_type.qparams.scales,
        output_type.qparams.scale,
        bounds,
    )
    return Conv2DStep(
        name=op.name,
        input=op.input,
        weight=op.weight,
        bias=op.bias,
        output=op.output,
        stride=op.stride,
        dilation=op.dilation,
        padding=op.padding,
        groups=op.groups,
        multipliers=multipliers,
        shifts=shifts,
        activation_min=op.activation_min,
        activation_max=op.activation_max,
        accumulator_bounds=bounds,
    )


@lower_op.register
def _lower_depthwise_conv2d(
    op: DepthwiseConv2DOp,
    graph: QuantizedGraph,
) -> DepthwiseConv2DStep:
    input_type = graph.values[op.input]
    weight_type = graph.values[op.weight]
    output_type = graph.values[op.output]
    assert isinstance(input_type.qparams, PerTensorQParams)
    assert isinstance(weight_type.qparams, PerAxisQParams)
    assert isinstance(output_type.qparams, PerTensorQParams)
    weight = graph.constants[op.weight]
    bias = graph.constants[op.bias]
    output_channels = weight.shape[2]
    bounds = _accumulator_bounds(
        input_type.qparams.zero_point,
        weight,
        bias,
        output_channels,
        2,
        op.name,
    )
    multipliers, shifts = _requantization(
        op.name,
        input_type.qparams.scale,
        weight_type.qparams.scales,
        output_type.qparams.scale,
        bounds,
    )
    return DepthwiseConv2DStep(
        name=op.name,
        input=op.input,
        weight=op.weight,
        bias=op.bias,
        output=op.output,
        depth_multiplier=op.depth_multiplier,
        stride=op.stride,
        dilation=op.dilation,
        padding=op.padding,
        multipliers=multipliers,
        shifts=shifts,
        activation_min=op.activation_min,
        activation_max=op.activation_max,
        accumulator_bounds=bounds,
    )


__all__: list[str] = []
