from __future__ import annotations

import math

import numpy as np

from bakenn.errors import GraphValidationError
from bakenn.ir.graph import QuantizedGraph
from bakenn.ir.ops.spatial import ConvTranspose2DOp, ResizeBilinear2DOp, ResizeNearest2DOp
from bakenn.ir.types import DType, Layout, PerAxisQParams, PerTensorQParams, normalize_scale_float32
from bakenn.ir.verify import verify_op


def _fail(op: object, message: str) -> None:
    raise GraphValidationError(f"{getattr(op, 'name')}: {message}")


def _verify_resize(op: ResizeNearest2DOp | ResizeBilinear2DOp, graph: QuantizedGraph) -> None:
    source = graph.values[op.input]
    output = graph.values[op.output]
    if source.dtype is not DType.INT8 or output.dtype is not DType.INT8:
        _fail(op, "Resize2D requires int8 tensors")
    if source.layout is not Layout.NHWC or output.layout is not Layout.NHWC:
        _fail(op, "Resize2D requires NHWC tensors")
    if len(source.shape) != 4 or len(output.shape) != 4 or source.shape[0] != 1 or output.shape[0] != 1:
        _fail(op, "Resize2D requires static rank-four batch-one tensors")
    if source.shape[3] != output.shape[3]:
        _fail(op, "Resize2D preserves channels")
    if output.shape[1] <= 0 or output.shape[2] <= 0:
        _fail(op, "Resize2D output spatial dimensions must be positive")
    if not isinstance(source.qparams, PerTensorQParams) or source.qparams != output.qparams:
        _fail(op, "Resize2D preserves per-tensor qparams")


@verify_op.register
def _verify_nearest(op: ResizeNearest2DOp, graph: QuantizedGraph) -> None:
    _verify_resize(op, graph)


@verify_op.register
def _verify_bilinear(op: ResizeBilinear2DOp, graph: QuantizedGraph) -> None:
    _verify_resize(op, graph)


@verify_op.register
def _verify_transpose(op: ConvTranspose2DOp, graph: QuantizedGraph) -> None:
    source = graph.values[op.input]
    weight = graph.values[op.weight]
    bias = graph.values[op.bias]
    output = graph.values[op.output]
    if op.weight not in graph.constants or op.bias not in graph.constants:
        _fail(op, "ConvTranspose2D weight and bias must be constants")
    if source.dtype is not DType.INT8 or output.dtype is not DType.INT8:
        _fail(op, "ConvTranspose2D activations must be int8")
    if source.layout is not Layout.NHWC or output.layout is not Layout.NHWC:
        _fail(op, "ConvTranspose2D activations must use NHWC")
    if len(source.shape) != 4 or len(output.shape) != 4 or source.shape[0] != 1 or output.shape[0] != 1:
        _fail(op, "ConvTranspose2D requires static rank-four batch-one activations")
    if not isinstance(source.qparams, PerTensorQParams) or not isinstance(output.qparams, PerTensorQParams):
        _fail(op, "ConvTranspose2D activations require per-tensor qparams")
    if weight.dtype is not DType.INT8 or weight.layout is not Layout.OHWI or len(weight.shape) != 4:
        _fail(op, "ConvTranspose2D weight must be OHWI int8")
    output_channels, kernel_h, kernel_w, input_channels_per_group = weight.shape
    input_channels = source.shape[3]
    if (
        input_channels % op.groups != 0
        or output_channels % op.groups != 0
        or input_channels_per_group != input_channels // op.groups
        or output.shape[3] != output_channels
    ):
        _fail(op, "ConvTranspose2D channels do not match OHWI weight")
    if bias.dtype is not DType.INT32 or bias.layout is not Layout.C or bias.shape != (output_channels,):
        _fail(op, "ConvTranspose2D bias must be one int32 value per output channel")
    if not isinstance(weight.qparams, PerAxisQParams) or weight.qparams.axis != 0 or len(weight.qparams.scales) != output_channels:
        _fail(op, "ConvTranspose2D weights require output-channel qparams on axis zero")
    if not isinstance(bias.qparams, PerAxisQParams) or bias.qparams.axis != 0:
        _fail(op, "ConvTranspose2D bias requires per-output-channel qparams")
    if any(weight.qparams.zero_points) or any(bias.qparams.zero_points):
        _fail(op, "ConvTranspose2D weight and bias zero-points must be zero")
    if np.any(graph.constants[op.weight] == -128):
        _fail(op, "symmetric ConvTranspose2D weights must stay in [-127, 127]")
    expected_bias_scales = tuple(
        normalize_scale_float32(source.qparams.scale * scale) for scale in weight.qparams.scales
    )
    if len(bias.qparams.scales) != output_channels or any(
        not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.0)
        for actual, expected in zip(bias.qparams.scales, expected_bias_scales)
    ):
        _fail(op, "ConvTranspose2D bias scale must equal input_scale * weight_scale[channel]")
    if any(value <= 0 for value in (*op.stride, *op.dilation)) or any(value < 0 for value in (*op.padding, *op.output_padding)):
        _fail(op, "ConvTranspose2D stride/dilation must be positive and padding nonnegative")
    if any(op.output_padding[axis] >= op.stride[axis] for axis in range(2)):
        _fail(op, "ConvTranspose2D output_padding must be smaller than stride")
    expected = (
        1,
        (source.shape[1] - 1) * op.stride[0] - op.padding[0] - op.padding[1]
        + op.dilation[0] * (kernel_h - 1) + op.output_padding[0] + 1,
        (source.shape[2] - 1) * op.stride[1] - op.padding[2] - op.padding[3]
        + op.dilation[1] * (kernel_w - 1) + op.output_padding[1] + 1,
        output_channels,
    )
    if output.shape != expected:
        _fail(op, f"ConvTranspose2D output shape {output.shape} does not match expected {expected}")


__all__: list[str] = []
