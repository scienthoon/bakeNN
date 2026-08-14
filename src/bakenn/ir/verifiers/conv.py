from __future__ import annotations

import math

import numpy as np

from bakenn.errors import GraphValidationError
from bakenn.ir.graph import QuantizedGraph
from bakenn.ir.ops.conv import Conv2DOp, DepthwiseConv2DOp
from bakenn.ir.types import (
    DType,
    Layout,
    PerAxisQParams,
    PerTensorQParams,
    TARGET_SIZE_MAX,
    normalize_scale_float32,
)
from bakenn.ir.verify import verify_op


def _fail(op: Conv2DOp | DepthwiseConv2DOp, message: str) -> None:
    raise GraphValidationError(f"{op.name}: {message}")


def _verify_common(
    op: Conv2DOp | DepthwiseConv2DOp,
    graph: QuantizedGraph,
) -> tuple[object, object, object, object]:
    x = graph.values[op.input]
    weight = graph.values[op.weight]
    bias = graph.values[op.bias]
    output = graph.values[op.output]

    if op.weight not in graph.constants or op.bias not in graph.constants:
        _fail(op, "convolution weight and bias must be compile-time constants")
    if x.dtype is not DType.INT8 or output.dtype is not DType.INT8:
        _fail(op, "convolution activations must be int8")
    if x.layout is not Layout.NHWC or output.layout is not Layout.NHWC:
        _fail(op, "convolution activations must use NHWC layout")
    if len(x.shape) != 4 or len(output.shape) != 4 or x.shape[0] != 1 or output.shape[0] != 1:
        _fail(op, "P0 convolution requires rank-four static batch size one")
    if not isinstance(x.qparams, PerTensorQParams) or not isinstance(
        output.qparams, PerTensorQParams
    ):
        _fail(op, "convolution activations require per-tensor qparams")

    if bias.dtype is not DType.INT32 or bias.layout is not Layout.C:
        _fail(op, "bias must use int32 C layout")
    if not isinstance(bias.qparams, PerAxisQParams) or bias.qparams.axis != 0:
        _fail(op, "bias requires per-output-channel qparams on axis zero")
    if any(zero_point != 0 for zero_point in bias.qparams.zero_points):
        _fail(op, "bias zero_points must be zero")

    if any(value <= 0 for value in (*op.stride, *op.dilation)):
        _fail(op, "stride and dilation must be positive")
    if any(value < 0 for value in op.padding):
        _fail(op, "explicit padding must be nonnegative")
    target_parameters = (*op.stride, *op.dilation, *op.padding)
    if any(value > TARGET_SIZE_MAX for value in target_parameters):
        _fail(op, "stride, dilation, and padding must fit the 32-bit target ABI")
    if not -128 <= op.activation_min <= op.activation_max <= 127:
        _fail(op, "invalid fused activation clamp")
    return x, weight, bias, output


def _verify_output_shape(
    op: Conv2DOp | DepthwiseConv2DOp,
    input_shape: tuple[int, ...],
    kernel_shape: tuple[int, int],
    output_shape: tuple[int, ...],
    output_channels: int,
) -> None:
    _, input_height, input_width, _ = input_shape
    kernel_height, kernel_width = kernel_shape
    stride_height, stride_width = op.stride
    dilation_height, dilation_width = op.dilation
    pad_top, pad_bottom, pad_left, pad_right = op.padding
    effective_height = dilation_height * (kernel_height - 1) + 1
    effective_width = dilation_width * (kernel_width - 1) + 1
    max_signed_coordinate = (1 << 63) - 1
    max_runtime_y = (output_shape[1] - 1) * stride_height + (
        kernel_height - 1
    ) * dilation_height
    max_runtime_x = (output_shape[2] - 1) * stride_width + (
        kernel_width - 1
    ) * dilation_width
    if max_runtime_y > max_signed_coordinate or max_runtime_x > max_signed_coordinate:
        _fail(op, "runtime convolution coordinates exceed int64")
    height_numerator = input_height + pad_top + pad_bottom - effective_height
    width_numerator = input_width + pad_left + pad_right - effective_width
    if height_numerator < 0 or width_numerator < 0:
        _fail(op, "effective kernel exceeds padded input")
    expected = (
        1,
        height_numerator // stride_height + 1,
        width_numerator // stride_width + 1,
        output_channels,
    )
    if output_shape != expected:
        _fail(op, f"output shape {output_shape} does not match expected {expected}")


def _verify_weight_and_bias_qparams(
    op: Conv2DOp | DepthwiseConv2DOp,
    graph: QuantizedGraph,
    weight: object,
    bias: object,
    input_scale: float,
    output_channels: int,
    weight_axis: int,
) -> None:
    weight_qparams = weight.qparams  # type: ignore[attr-defined]
    bias_qparams = bias.qparams  # type: ignore[attr-defined]
    if not isinstance(weight_qparams, PerAxisQParams):
        _fail(op, "weight requires per-output-channel qparams")
    if weight_qparams.axis != weight_axis or len(weight_qparams.scales) != output_channels:
        _fail(op, f"weight qparams must have {output_channels} channels on axis {weight_axis}")
    if any(zero_point != 0 for zero_point in weight_qparams.zero_points):
        _fail(op, "symmetric weights require zero_point zero")
    if np.any(graph.constants[op.weight] == -128):
        _fail(op, "symmetric weights must stay in [-127, 127]")
    if bias.shape != (output_channels,):  # type: ignore[attr-defined]
        _fail(op, "bias must contain one value per output channel")
    try:
        expected_scales = tuple(
            normalize_scale_float32(input_scale * scale)
            for scale in weight_qparams.scales
        )
    except ValueError:
        _fail(op, "bias scale product is not representable as float32")
    if len(bias_qparams.scales) != output_channels or any(
        not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.0)
        for actual, expected in zip(bias_qparams.scales, expected_scales)
    ):
        _fail(op, "bias scale must equal input_scale * weight_scale[channel]")


@verify_op.register
def _verify_conv2d(op: Conv2DOp, graph: QuantizedGraph) -> None:
    x, weight, bias, output = _verify_common(op, graph)
    if op.groups <= 0 or op.groups > TARGET_SIZE_MAX:
        _fail(op, "Conv2D groups must be positive and fit the 32-bit target ABI")
    if weight.dtype is not DType.INT8 or weight.layout is not Layout.OHWI or len(weight.shape) != 4:
        _fail(op, "Conv2D weight must be rank-four OHWI int8")
    output_channels, kernel_height, kernel_width, weight_input_channels = weight.shape
    if x.shape[3] % op.groups or output_channels % op.groups:
        _fail(op, "Conv2D groups must divide both input and output channels")
    expected_weight_channels = x.shape[3] // op.groups
    if weight_input_channels != expected_weight_channels:
        _fail(
            op,
            "Conv2D OHWI input channels must equal activation channels / groups",
        )
    if output.shape[3] != output_channels:
        _fail(op, "Conv2D output channels do not match weight channels")
    _verify_weight_and_bias_qparams(
        op, graph, weight, bias, x.qparams.scale, output_channels, weight_axis=0
    )
    _verify_output_shape(
        op,
        x.shape,
        (kernel_height, kernel_width),
        output.shape,
        output_channels,
    )


@verify_op.register
def _verify_depthwise_conv2d(op: DepthwiseConv2DOp, graph: QuantizedGraph) -> None:
    x, weight, bias, output = _verify_common(op, graph)
    if op.depth_multiplier <= 0:
        _fail(op, "depth_multiplier must be positive")
    if op.depth_multiplier > TARGET_SIZE_MAX:
        _fail(op, "depth_multiplier must fit the 32-bit target ABI")
    if weight.dtype is not DType.INT8 or weight.layout is not Layout.HWO or len(weight.shape) != 3:
        _fail(op, "DepthwiseConv2D weight must be rank-three HWO int8")
    kernel_height, kernel_width, output_channels = weight.shape
    expected_channels = x.shape[3] * op.depth_multiplier
    if output_channels != expected_channels or output.shape[3] != expected_channels:
        _fail(op, "C_out must equal C_in * depth_multiplier")
    _verify_weight_and_bias_qparams(
        op, graph, weight, bias, x.qparams.scale, output_channels, weight_axis=2
    )
    _verify_output_shape(
        op,
        x.shape,
        (kernel_height, kernel_width),
        output.shape,
        output_channels,
    )


__all__: list[str] = []
