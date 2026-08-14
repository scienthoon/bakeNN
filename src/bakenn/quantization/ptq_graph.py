from __future__ import annotations

import math
from typing import Iterator, Mapping

import numpy as np

from bakenn.errors import CompileError
from bakenn.frontends.torch_export.model import (
    FloatAddOp,
    FloatAveragePool1DOp,
    FloatAveragePool2DOp,
    FloatConcatOp,
    FloatConv1DOp,
    FloatConv2DOp,
    FloatConvTranspose2DOp,
    FloatDepthwiseConv2DOp,
    FloatFlattenOp,
    FloatGraph,
    FloatHardSigmoidOp,
    FloatHardSwishOp,
    FloatLayout,
    FloatLinearOp,
    FloatMaxPool2DOp,
    FloatMaxPool1DOp,
    FloatMulOp,
    FloatPad2DOp,
    FloatReLU6Op,
    FloatReLUOp,
    FloatReshapeOp,
    FloatReduceMeanOp,
    FloatResizeBilinear2DOp,
    FloatResizeNearest2DOp,
    FloatSigmoidOp,
    FloatSiLUOp,
    FloatSliceOp,
    FloatSoftmaxOp,
    FloatValueKind,
)
from bakenn.ir.graph import QuantizedGraph
from bakenn.ir.op import LinearOp
from bakenn.ir.ops.conv import Conv2DOp, DepthwiseConv2DOp
from bakenn.ir.ops.activation import HardSigmoidOp, HardSwishOp, SiLUOp, SigmoidOp
from bakenn.ir.ops.elementwise import AddOp, ClampOp, MulOp, RequantizeOp
from bakenn.ir.ops.pool import AveragePool2DOp, MaxPool2DOp
from bakenn.ir.ops.sequence import AveragePool1DOp, Conv1DOp, MaxPool1DOp
from bakenn.ir.ops.shape import ConcatenateOp, FlattenOp, ReshapeOp, SliceOp
from bakenn.ir.ops.softmax import SoftmaxOp
from bakenn.ir.ops.tensor import Pad2DOp, ReduceMeanOp
from bakenn.ir.ops.spatial import ConvTranspose2DOp, ResizeBilinear2DOp, ResizeNearest2DOp
from bakenn.ir.types import (
    DType,
    Layout,
    PerTensorQParams,
    TensorType,
    normalize_scale_float32,
)
from bakenn.ir.verify import verify_graph
from bakenn.ir.verifiers.softmax import SOFTMAX_OUTPUT_SCALE, SOFTMAX_OUTPUT_ZERO_POINT
from bakenn.passes import fuse_clamps, legalize_graph
from bakenn.quantization.fixedpoint import ARITHMETIC_PROFILE
from bakenn.quantization.primitives import quantize_compute_constants


def _round_away(values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.where(array >= 0.0, np.floor(array + 0.5), np.ceil(array - 0.5))


def _activation_qparams(minimum: float, maximum: float) -> PerTensorQParams:
    minimum = min(float(minimum), 0.0)
    maximum = max(float(maximum), 0.0)
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise CompileError("calibration produced NaN or infinity")
    if minimum == maximum:
        return PerTensorQParams(1.0, 0)
    raw_scale = (maximum - minimum) / 255.0
    try:
        scale = normalize_scale_float32(raw_scale)
    except ValueError as error:
        raise CompileError("calibration range cannot be represented by INT8") from error
    zero_point = int(_round_away(-128.0 - minimum / scale).item())
    return PerTensorQParams(scale, max(-128, min(127, zero_point)))


def _quantize_per_tensor(array: np.ndarray, qparams: PerTensorQParams) -> np.ndarray:
    quantized = _round_away(np.asarray(array, dtype=np.float64) / qparams.scale)
    quantized += qparams.zero_point
    return np.ascontiguousarray(np.clip(quantized, -128, 127).astype(np.int8))


def _quantize_code(real_value: float, qparams: PerTensorQParams) -> int:
    code = int(_round_away(real_value / qparams.scale).item()) + qparams.zero_point
    return max(-128, min(127, code))


def _conv2d(
    input_value: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    stride: tuple[int, int],
    padding: tuple[int, int, int, int],
    dilation: tuple[int, int],
    *,
    depthwise: bool,
    depth_multiplier: int = 1,
    groups: int = 1,
) -> np.ndarray:
    _, input_channels, input_height, input_width = input_value.shape
    output_channels, _, kernel_height, kernel_width = weight.shape
    effective_height = dilation[0] * (kernel_height - 1) + 1
    effective_width = dilation[1] * (kernel_width - 1) + 1
    output_height = (
        input_height + padding[0] + padding[1] - effective_height
    ) // stride[0] + 1
    output_width = (
        input_width + padding[2] + padding[3] - effective_width
    ) // stride[1] + 1
    result = np.empty((1, output_channels, output_height, output_width), dtype=np.float32)
    for output_channel in range(output_channels):
        if depthwise:
            input_channel_start = output_channel // depth_multiplier
            input_channel_end = input_channel_start + 1
            group_input_channels = 1
        else:
            output_channels_per_group = output_channels // groups
            group_input_channels = input_channels // groups
            input_channel_start = (output_channel // output_channels_per_group) * group_input_channels
            input_channel_end = input_channel_start + group_input_channels
        for output_y in range(output_height):
            origin_y = output_y * stride[0] - padding[0]
            for output_x in range(output_width):
                origin_x = output_x * stride[1] - padding[2]
                accumulator = np.float32(0.0 if bias is None else bias[output_channel])
                for input_channel in range(input_channel_start, input_channel_end):
                    weight_channel = 0 if depthwise else input_channel - input_channel_start
                    for kernel_y in range(kernel_height):
                        input_y = origin_y + kernel_y * dilation[0]
                        if not 0 <= input_y < input_height:
                            continue
                        for kernel_x in range(kernel_width):
                            input_x = origin_x + kernel_x * dilation[1]
                            if not 0 <= input_x < input_width:
                                continue
                            accumulator = np.float32(
                                accumulator
                                + input_value[0, input_channel, input_y, input_x]
                                * weight[output_channel, weight_channel, kernel_y, kernel_x]
                            )
                result[0, output_channel, output_y, output_x] = accumulator
    return result


def _pool2d(
    input_value: np.ndarray,
    kernel: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int, int, int],
    *,
    average: bool,
) -> np.ndarray:
    _, channels, input_height, input_width = input_value.shape
    output_height = (input_height + padding[0] + padding[1] - kernel[0]) // stride[0] + 1
    output_width = (input_width + padding[2] + padding[3] - kernel[1]) // stride[1] + 1
    result = np.empty((1, channels, output_height, output_width), dtype=np.float32)
    for channel in range(channels):
        for output_y in range(output_height):
            start_y = output_y * stride[0] - padding[0]
            for output_x in range(output_width):
                start_x = output_x * stride[1] - padding[2]
                samples: list[np.float32] = []
                for kernel_y in range(kernel[0]):
                    input_y = start_y + kernel_y
                    if not 0 <= input_y < input_height:
                        continue
                    for kernel_x in range(kernel[1]):
                        input_x = start_x + kernel_x
                        if 0 <= input_x < input_width:
                            samples.append(input_value[0, channel, input_y, input_x])
                if not samples:
                    raise CompileError("pooling window has no valid input elements")
                if average:
                    result[0, channel, output_y, output_x] = np.mean(
                        np.asarray(samples, dtype=np.float32), dtype=np.float32
                    )
                else:
                    result[0, channel, output_y, output_x] = np.max(samples)
    return result


def _conv_transpose2d(
    input_value: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    stride: tuple[int, int],
    padding: tuple[int, int, int, int],
    output_padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
) -> np.ndarray:
    _, input_channels, input_height, input_width = input_value.shape
    weight_input_channels, output_channels_per_group, kernel_height, kernel_width = weight.shape
    if weight_input_channels != input_channels:
        raise CompileError("ConvTranspose2D input channels do not match weight")
    if input_channels % groups != 0:
        raise CompileError("ConvTranspose2D input channels must be divisible by groups")
    output_channels = output_channels_per_group * groups
    input_channels_per_group = input_channels // groups
    output_height = (
        (input_height - 1) * stride[0] - padding[0] - padding[1]
        + dilation[0] * (kernel_height - 1) + output_padding[0] + 1
    )
    output_width = (
        (input_width - 1) * stride[1] - padding[2] - padding[3]
        + dilation[1] * (kernel_width - 1) + output_padding[1] + 1
    )
    result = np.empty((1, output_channels, output_height, output_width), dtype=np.float32)
    for output_channel in range(output_channels):
        result[0, output_channel, :, :] = np.float32(
            0.0 if bias is None else bias[output_channel]
        )
    for input_channel in range(input_channels):
        for input_y in range(input_height):
            for input_x in range(input_width):
                source = input_value[0, input_channel, input_y, input_x]
                group = input_channel // input_channels_per_group
                output_start = group * output_channels_per_group
                for local_output_channel in range(output_channels_per_group):
                    output_channel = output_start + local_output_channel
                    for kernel_y in range(kernel_height):
                        output_y = input_y * stride[0] - padding[0] + kernel_y * dilation[0]
                        if not 0 <= output_y < output_height:
                            continue
                        for kernel_x in range(kernel_width):
                            output_x = input_x * stride[1] - padding[2] + kernel_x * dilation[1]
                            if 0 <= output_x < output_width:
                                result[0, output_channel, output_y, output_x] = np.float32(
                                    result[0, output_channel, output_y, output_x]
                                    + source * weight[input_channel, local_output_channel, kernel_y, kernel_x]
                                )
    return result


def _resize2d(input_value: np.ndarray, output_shape: tuple[int, ...], *, bilinear: bool, align_corners: bool = False) -> np.ndarray:
    _, channels, input_height, input_width = input_value.shape
    _, _, output_height, output_width = output_shape
    result = np.empty(output_shape, dtype=np.float32)
    for output_y in range(output_height):
        if align_corners and output_height > 1:
            source_y = output_y * (input_height - 1) / (output_height - 1)
        else:
            source_y = max(0.0, min(input_height - 1.0, (output_y + 0.5) * input_height / output_height - 0.5))
        y0 = int(math.floor(source_y))
        y1 = min(input_height - 1, y0 + 1)
        wy = source_y - y0
        for output_x in range(output_width):
            if not bilinear:
                source_x = output_x * input_width // output_width
                result[0, :, output_y, output_x] = input_value[0, :, output_y * input_height // output_height, source_x]
                continue
            if align_corners and output_width > 1:
                source_x_f = output_x * (input_width - 1) / (output_width - 1)
            else:
                source_x_f = max(0.0, min(input_width - 1.0, (output_x + 0.5) * input_width / output_width - 0.5))
            x0 = int(math.floor(source_x_f))
            x1 = min(input_width - 1, x0 + 1)
            wx = source_x_f - x0
            result[0, :, output_y, output_x] = (
                input_value[0, :, y0, x0] * (1.0 - wy) * (1.0 - wx)
                + input_value[0, :, y0, x1] * (1.0 - wy) * wx
                + input_value[0, :, y1, x0] * wy * (1.0 - wx)
                + input_value[0, :, y1, x1] * wy * wx
            )
    return result


def _conv1d(
    input_value: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    stride: int,
    padding: tuple[int, int],
    dilation: int,
    groups: int,
) -> np.ndarray:
    _, input_channels, input_length = input_value.shape
    output_channels, group_input_channels, kernel = weight.shape
    output_length = (input_length + sum(padding) - dilation * (kernel - 1) - 1) // stride + 1
    output_channels_per_group = output_channels // groups
    result = np.empty((1, output_channels, output_length), dtype=np.float32)
    for output_channel in range(output_channels):
        group = output_channel // output_channels_per_group
        input_base = group * group_input_channels
        for position in range(output_length):
            accumulator = np.float32(0.0 if bias is None else bias[output_channel])
            for kernel_index in range(kernel):
                source_position = position * stride + kernel_index * dilation - padding[0]
                if not 0 <= source_position < input_length:
                    continue
                for local_channel in range(group_input_channels):
                    accumulator = np.float32(
                        accumulator
                        + input_value[0, input_base + local_channel, source_position]
                        * weight[output_channel, local_channel, kernel_index]
                    )
            result[0, output_channel, position] = accumulator
    return result


def _pool1d(
    input_value: np.ndarray,
    kernel: int,
    stride: int,
    padding: tuple[int, int],
    *,
    average: bool,
) -> np.ndarray:
    _, channels, input_length = input_value.shape
    output_length = (input_length + sum(padding) - kernel) // stride + 1
    result = np.empty((1, channels, output_length), dtype=np.float32)
    for channel in range(channels):
        for position in range(output_length):
            start = position * stride - padding[0]
            samples = input_value[0, channel, max(0, start):min(input_length, start + kernel)]
            if samples.size == 0:
                raise CompileError("Pool1D window has no valid input elements")
            result[0, channel, position] = (
                np.mean(samples, dtype=np.float32) if average else np.max(samples)
            )
    return result


def _evaluate(graph: FloatGraph, input_value: np.ndarray) -> dict[str, np.ndarray]:
    expected = graph.values[graph.inputs[0]].shape
    source = np.asarray(input_value, dtype=np.float32)
    if source.shape != expected:
        raise CompileError(f"calibration sample shape {source.shape} does not match {expected}")
    if not np.all(np.isfinite(source)):
        raise CompileError("calibration data contains NaN or infinity")
    values: dict[str, np.ndarray] = {
        graph.inputs[0]: np.ascontiguousarray(source),
        **{name: np.asarray(value, dtype=np.float32) for name, value in graph.constants.items()},
    }
    for op in graph.ops:
        if isinstance(op, FloatConv1DOp):
            output = _conv1d(
                values[op.input],
                values[op.weight],
                None if op.bias is None else values[op.bias],
                op.stride,
                op.padding,
                op.dilation,
                op.groups,
            )
        elif isinstance(op, FloatConv2DOp):
            output = _conv2d(
                values[op.input],
                values[op.weight],
                None if op.bias is None else values[op.bias],
                op.stride,
                op.padding,
                op.dilation,
                depthwise=False,
                groups=op.groups,
            )
        elif isinstance(op, FloatDepthwiseConv2DOp):
            output = _conv2d(
                values[op.input],
                values[op.weight],
                None if op.bias is None else values[op.bias],
                op.stride,
                op.padding,
                op.dilation,
                depthwise=True,
                depth_multiplier=op.depth_multiplier,
            )
        elif isinstance(op, FloatConvTranspose2DOp):
            output = _conv_transpose2d(
                values[op.input],
                values[op.weight],
                None if op.bias is None else values[op.bias],
                op.stride,
                op.padding,
                op.output_padding,
                op.dilation,
                op.groups,
            )
        elif isinstance(op, FloatLinearOp):
            output = values[op.input] @ values[op.weight].T
            if op.bias is not None:
                output = output + values[op.bias]
        elif isinstance(op, FloatAddOp):
            output = values[op.input_a] + values[op.input_b]
        elif isinstance(op, FloatMulOp):
            output = values[op.input_a] * values[op.input_b]
        elif isinstance(op, FloatSigmoidOp):
            source = values[op.input].astype(np.float64)
            output = np.where(
                source >= 0.0,
                1.0 / (1.0 + np.exp(-source)),
                np.exp(source) / (1.0 + np.exp(source)),
            )
        elif isinstance(op, FloatHardSigmoidOp):
            source = values[op.input]
            output = np.clip(source + 3.0, 0.0, 6.0) / 6.0
        elif isinstance(op, FloatHardSwishOp):
            source = values[op.input]
            output = source * np.clip(source + 3.0, 0.0, 6.0) / 6.0
        elif isinstance(op, FloatSiLUOp):
            source = values[op.input].astype(np.float64)
            sigmoid = np.where(
                source >= 0.0,
                1.0 / (1.0 + np.exp(-source)),
                np.exp(source) / (1.0 + np.exp(source)),
            )
            output = source * sigmoid
        elif isinstance(op, FloatReLU6Op):
            output = np.clip(values[op.input], 0.0, 6.0)
        elif isinstance(op, FloatReLUOp):
            output = np.maximum(values[op.input], 0.0)
        elif isinstance(op, FloatAveragePool2DOp):
            output = _pool2d(values[op.input], op.kernel, op.stride, op.padding, average=True)
        elif isinstance(op, FloatMaxPool2DOp):
            output = _pool2d(values[op.input], op.kernel, op.stride, op.padding, average=False)
        elif isinstance(op, FloatAveragePool1DOp):
            output = _pool1d(values[op.input], op.kernel, op.stride, op.padding, average=True)
        elif isinstance(op, FloatMaxPool1DOp):
            output = _pool1d(values[op.input], op.kernel, op.stride, op.padding, average=False)
        elif isinstance(op, FloatPad2DOp):
            top, bottom, left, right = op.padding
            output = np.pad(
                values[op.input],
                ((0, 0), (0, 0), (top, bottom), (left, right)),
                mode="constant",
                constant_values=0.0,
            )
        elif isinstance(op, FloatReduceMeanOp):
            output = np.mean(values[op.input], axis=op.axes, keepdims=op.keepdims, dtype=np.float32)
        elif isinstance(op, FloatResizeNearest2DOp):
            output = _resize2d(values[op.input], graph.values[op.output].shape, bilinear=False)
        elif isinstance(op, FloatResizeBilinear2DOp):
            output = _resize2d(
                values[op.input], graph.values[op.output].shape,
                bilinear=True, align_corners=op.align_corners,
            )
        elif isinstance(op, FloatFlattenOp):
            output = values[op.input].reshape(graph.values[op.output].shape)
        elif isinstance(op, FloatReshapeOp):
            output = values[op.input].reshape(op.target_shape)
        elif isinstance(op, FloatSliceOp):
            selection = [slice(None)] * values[op.input].ndim
            selection[op.axis] = slice(op.start, op.stop, op.step)
            output = values[op.input][tuple(selection)]
        elif isinstance(op, FloatConcatOp):
            output = np.concatenate([values[name] for name in op.input_values], axis=op.axis)
        elif isinstance(op, FloatSoftmaxOp):
            shifted = values[op.input] - np.max(values[op.input], axis=-1, keepdims=True)
            exponentials = np.exp(shifted.astype(np.float64))
            output = exponentials / np.sum(exponentials, axis=-1, keepdims=True)
        else:
            raise CompileError(
                f"{op.name}: PTQ evaluator has no implementation for {type(op).__name__}"
            )
        result = np.ascontiguousarray(np.asarray(output, dtype=np.float32))
        expected_shape = graph.values[op.output].shape
        if result.shape != expected_shape:
            raise CompileError(
                f"{op.name}: FP32 evaluator produced {result.shape}, expected {expected_shape}"
            )
        if not np.all(np.isfinite(result)):
            raise CompileError(f"{op.name}: FP32 evaluator produced NaN or infinity")
        values[op.output] = result
    return values


def _samples(calibration_data: object, expected_shape: tuple[int, ...]) -> Iterator[np.ndarray]:
    def is_array_field(value: object) -> bool:
        return isinstance(value, np.ndarray) or hasattr(value, "detach")

    def array(value: object) -> np.ndarray:
        if isinstance(value, (list, tuple)):
            if not value:
                raise CompileError("calibration batch container is empty")
            if any(is_array_field(item) for item in value):
                if len(value) != 1:
                    raise CompileError(
                        "calibration batch has multiple fields; (input, target) batches are "
                        "ambiguous, provide input tensors only"
                    )
                value = value[0]
        if hasattr(value, "detach"):
            try:
                value = value.detach().cpu().numpy()
            except (AttributeError, RuntimeError, TypeError, ValueError) as error:
                raise CompileError(
                    "calibration tensor cannot be converted through the array protocol"
                ) from error
        try:
            result = np.asarray(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise CompileError("calibration samples must be numeric tensors or arrays") from error
        if result.dtype.kind not in "iuf":
            raise CompileError("calibration samples must use a real numeric dtype")
        return result

    def split(value: np.ndarray) -> Iterator[np.ndarray]:
        if value.shape == expected_shape:
            yield value
            return
        element_shape = expected_shape[1:]
        if value.shape == element_shape:
            yield value.reshape(expected_shape)
            return
        if value.ndim == len(expected_shape) and value.shape[1:] == element_shape:
            for index in range(value.shape[0]):
                yield value[index : index + 1]
            return
        raise CompileError(
            f"calibration shape {value.shape} is incompatible with batch-one input {expected_shape}"
        )

    def snapshot(value: np.ndarray) -> np.ndarray:
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                result = np.array(value, dtype=np.float32, copy=True, order="C")
        except (TypeError, ValueError, OverflowError) as error:
            raise CompileError("calibration samples cannot be represented as FP32") from error
        if not np.all(np.isfinite(result)):
            raise CompileError("calibration data contains NaN or infinity")
        result.setflags(write=False)
        return result

    if isinstance(calibration_data, np.ndarray) or hasattr(calibration_data, "detach"):
        batches: Iterator[object] = iter((calibration_data,))
    else:
        try:
            batches = iter(calibration_data)  # type: ignore[arg-type]
        except TypeError as error:
            raise CompileError("calibration_data must be an array or iterable") from error
    sample_count = 0
    for batch in batches:
        for sample in split(array(batch)):
            sample_count += 1
            # Snapshot before advancing the source iterator: zero-copy loaders
            # are allowed to reuse and mutate their backing batch buffer.
            yield snapshot(sample)
    if sample_count == 0:
        raise CompileError("calibration_data must contain at least one sample")


def _canonical_shape_layout(shape: tuple[int, ...]) -> tuple[tuple[int, ...], Layout]:
    if len(shape) == 4:
        return (shape[0], shape[2], shape[3], shape[1]), Layout.NHWC
    if len(shape) == 3:
        return (shape[0], shape[2], shape[1]), Layout.NLC
    if len(shape) == 2:
        return shape, Layout.NC
    raise CompileError(f"PTQ cannot canonicalize rank-{len(shape)} activation")


def _canonical_activation(array: np.ndarray) -> np.ndarray:
    if array.ndim == 4:
        return np.ascontiguousarray(array.transpose(0, 2, 3, 1))
    if array.ndim == 3:
        return np.ascontiguousarray(array.transpose(0, 2, 1))
    return np.ascontiguousarray(array)


def _consumers(graph: FloatGraph) -> Mapping[str, tuple[object, ...]]:
    result: dict[str, list[object]] = {}
    for op in graph.ops:
        for input_name in op.inputs:
            result.setdefault(input_name, []).append(op)
    return {name: tuple(items) for name, items in result.items()}


def quantize_float_graph(
    graph: FloatGraph,
    calibration_data: object,
    *,
    name: str | None = None,
) -> QuantizedGraph:
    """Deterministically PTQ a captured static FloatGraph into P0 INT8 IR."""

    if not isinstance(graph, FloatGraph):
        raise CompileError("quantize_float_graph requires a FloatGraph")
    if name is not None and (not isinstance(name, str) or not name):
        raise CompileError("quantized graph name must be a non-empty string")
    observed: dict[str, list[float]] = {
        value_name: [math.inf, -math.inf]
        for value_name, value in graph.values.items()
        if value.kind in (FloatValueKind.INPUT, FloatValueKind.ACTIVATION)
    }
    for sample in _samples(calibration_data, graph.values[graph.inputs[0]].shape):
        evaluated = _evaluate(graph, sample)
        for value_name in observed:
            value = evaluated[value_name]
            observed[value_name][0] = min(observed[value_name][0], float(np.min(value)))
            observed[value_name][1] = max(observed[value_name][1], float(np.max(value)))

    qparams: dict[str, PerTensorQParams] = {
        value_name: _activation_qparams(bounds[0], bounds[1])
        for value_name, bounds in observed.items()
    }
    for op in graph.ops:
        if isinstance(op, FloatSoftmaxOp):
            qparams[op.output] = PerTensorQParams(
                SOFTMAX_OUTPUT_SCALE, SOFTMAX_OUTPUT_ZERO_POINT
            )
        elif isinstance(op, (FloatSigmoidOp, FloatHardSigmoidOp)):
            qparams[op.output] = PerTensorQParams(1.0 / 256.0, -128)
        elif isinstance(op, (FloatResizeNearest2DOp, FloatResizeBilinear2DOp)):
            qparams[op.output] = qparams[op.input]
        elif isinstance(op, FloatSliceOp):
            qparams[op.output] = qparams[op.input]

    consumer_map = _consumers(graph)
    producer_by_output = {op.output: op for op in graph.ops}
    fusible_float_producers = (
        FloatConv1DOp,
        FloatConv2DOp,
        FloatDepthwiseConv2DOp,
        FloatLinearOp,
        FloatAddOp,
        FloatMulOp,
        FloatAveragePool2DOp,
        FloatMaxPool2DOp,
        FloatAveragePool1DOp,
        FloatMaxPool1DOp,
        FloatReduceMeanOp,
    )
    # A single-consumer activation edge is not externally observable. Giving
    # its producer the downstream ReLU domain preserves the declared rounding
    # point while enabling the typed Clamp to fuse into the producer.
    for value_name, consumers in consumer_map.items():
        if value_name in graph.outputs or len(consumers) != 1:
            continue
        consumer = consumers[0]
        producer = producer_by_output.get(value_name)
        if isinstance(consumer, (FloatReLUOp, FloatReLU6Op)) and isinstance(
            producer, fusible_float_producers
        ):
            qparams[value_name] = qparams[consumer.output]

    values: dict[str, TensorType] = {}
    constants: dict[str, np.ndarray] = {}
    operations: list[object] = []
    generic_constant_names: set[str] = set()
    specialized_constants: set[str] = set()
    for op in graph.ops:
        if isinstance(
            op,
            (
                FloatConv1DOp,
                FloatConv2DOp,
                FloatConvTranspose2DOp,
                FloatDepthwiseConv2DOp,
                FloatLinearOp,
            ),
        ):
            specialized_constants.add(op.weight)
            if op.bias is not None:
                specialized_constants.add(op.bias)
    for op in graph.ops:
        for input_name in op.inputs:
            if input_name in graph.constants and input_name not in specialized_constants:
                generic_constant_names.add(input_name)

    for value_name in observed:
        shape, layout = _canonical_shape_layout(graph.values[value_name].shape)
        values[value_name] = TensorType(shape, DType.INT8, layout, qparams[value_name])
    for constant_name in sorted(generic_constant_names):
        source = graph.constants[constant_name]
        constant_qparams = _activation_qparams(float(np.min(source)), float(np.max(source)))
        canonical = _canonical_activation(source)
        shape, layout = _canonical_shape_layout(tuple(source.shape))
        values[constant_name] = TensorType(shape, DType.INT8, layout, constant_qparams)
        constants[constant_name] = _quantize_per_tensor(canonical, constant_qparams)

    flattened_permutations: dict[str, np.ndarray] = {}

    def unique_value(base: str) -> str:
        candidate = base
        suffix = 1
        while candidate in values:
            candidate = f"{base}.{suffix}"
            suffix += 1
        return candidate

    def requantized_input(input_name: str, output_qparams: PerTensorQParams, op_name: str) -> str:
        if values[input_name].qparams == output_qparams:
            return input_name
        intermediate = unique_value(f"{input_name}.requantized_for.{op_name}")
        input_type = values[input_name]
        values[intermediate] = TensorType(
            input_type.shape, input_type.dtype, input_type.layout, output_qparams
        )
        operations.append(RequantizeOp(f"{op_name}.input_requantize", input_name, intermediate))
        return intermediate

    def quantized_compute_constants(
        op: FloatConv1DOp | FloatConv2DOp | FloatConvTranspose2DOp | FloatDepthwiseConv2DOp | FloatLinearOp,
        canonical_weight: np.ndarray,
        input_qparams: PerTensorQParams,
        *,
        weight_layout: Layout,
    ) -> tuple[str, str]:
        output_axis = 2 if weight_layout is Layout.HWO else 0
        output_channels = canonical_weight.shape[output_axis]
        bias_source = (
            np.zeros(output_channels, dtype=np.float32)
            if op.bias is None
            else np.asarray(graph.constants[op.bias], dtype=np.float32)
        )
        output_qparams = values[op.output].qparams
        assert isinstance(output_qparams, PerTensorQParams)
        quantized_weight, quantized_bias = quantize_compute_constants(
            canonical_weight,
            bias_source,
            layout=weight_layout,
            input_qparams=input_qparams,
            output_qparams=output_qparams,
        )
        weight_name = unique_value(f"{op.name}.weight")
        values[weight_name] = TensorType(
            tuple(quantized_weight.values.shape),
            DType.INT8,
            weight_layout,
            quantized_weight.qparams,
        )
        constants[weight_name] = np.ascontiguousarray(quantized_weight.values)

        bias_name = unique_value(f"{op.name}.bias")
        values[bias_name] = TensorType(
            (output_channels,),
            DType.INT32,
            Layout.C,
            quantized_bias.qparams,
        )
        constants[bias_name] = np.ascontiguousarray(quantized_bias.values)
        return weight_name, bias_name

    for op in graph.ops:
        if isinstance(op, FloatConv1DOp):
            input_qparams = values[op.input].qparams
            assert isinstance(input_qparams, PerTensorQParams)
            weight_name, bias_name = quantized_compute_constants(
                op,
                np.ascontiguousarray(graph.constants[op.weight].transpose(0, 2, 1)),
                input_qparams,
                weight_layout=Layout.OWI,
            )
            operations.append(
                Conv1DOp(
                    op.name,
                    op.input,
                    weight_name,
                    bias_name,
                    op.output,
                    stride=op.stride,
                    dilation=op.dilation,
                    padding=op.padding,
                    groups=op.groups,
                )
            )
        elif isinstance(op, FloatConv2DOp):
            input_qparams = values[op.input].qparams
            assert isinstance(input_qparams, PerTensorQParams)
            weight_name, bias_name = quantized_compute_constants(
                op,
                np.ascontiguousarray(graph.constants[op.weight].transpose(0, 2, 3, 1)),
                input_qparams,
                weight_layout=Layout.OHWI,
            )
            operations.append(
                Conv2DOp(
                    op.name,
                    op.input,
                    weight_name,
                    bias_name,
                    op.output,
                    stride=op.stride,
                    dilation=op.dilation,
                    padding=op.padding,
                    groups=op.groups,
                )
            )
        elif isinstance(op, FloatConvTranspose2DOp):
            input_qparams = values[op.input].qparams
            assert isinstance(input_qparams, PerTensorQParams)
            source_weight = graph.constants[op.weight]
            input_channels = source_weight.shape[0]
            output_channels_per_group = source_weight.shape[1]
            if input_channels % op.groups != 0:
                raise CompileError(
                    f"{op.name}: ConvTranspose2D input channels must be divisible by groups"
                )
            input_channels_per_group = input_channels // op.groups
            canonical_weight = np.ascontiguousarray(
                source_weight.reshape(
                    op.groups,
                    input_channels_per_group,
                    output_channels_per_group,
                    source_weight.shape[2],
                    source_weight.shape[3],
                )
                .transpose(0, 2, 3, 4, 1)
                .reshape(
                    op.groups * output_channels_per_group,
                    source_weight.shape[2],
                    source_weight.shape[3],
                    input_channels_per_group,
                )
            )
            weight_name, bias_name = quantized_compute_constants(
                op,
                canonical_weight,
                input_qparams,
                weight_layout=Layout.OHWI,
            )
            operations.append(
                ConvTranspose2DOp(
                    op.name,
                    op.input,
                    weight_name,
                    bias_name,
                    op.output,
                    stride=op.stride,
                    dilation=op.dilation,
                    padding=op.padding,
                    output_padding=op.output_padding,
                    groups=op.groups,
                )
            )
        elif isinstance(op, FloatDepthwiseConv2DOp):
            input_qparams = values[op.input].qparams
            assert isinstance(input_qparams, PerTensorQParams)
            weight_name, bias_name = quantized_compute_constants(
                op,
                np.ascontiguousarray(
                    graph.constants[op.weight][:, 0, :, :].transpose(1, 2, 0)
                ),
                input_qparams,
                weight_layout=Layout.HWO,
            )
            operations.append(
                DepthwiseConv2DOp(
                    op.name,
                    op.input,
                    weight_name,
                    bias_name,
                    op.output,
                    depth_multiplier=op.depth_multiplier,
                    stride=op.stride,
                    dilation=op.dilation,
                    padding=op.padding,
                )
            )
        elif isinstance(op, FloatLinearOp):
            input_qparams = values[op.input].qparams
            assert isinstance(input_qparams, PerTensorQParams)
            source_weight = np.asarray(graph.constants[op.weight], dtype=np.float32)
            permutation = flattened_permutations.get(op.input)
            canonical_weight = (
                source_weight
                if permutation is None
                else np.ascontiguousarray(source_weight[:, permutation])
            )

            weight_name, bias_name = quantized_compute_constants(
                op,
                canonical_weight,
                input_qparams,
                weight_layout=Layout.OI,
            )
            operations.append(LinearOp(op.name, op.input, weight_name, bias_name, op.output))
        elif isinstance(op, FloatAddOp):
            operations.append(AddOp(op.name, op.input_a, op.input_b, op.output))
        elif isinstance(op, FloatMulOp):
            operations.append(MulOp(op.name, op.input_a, op.input_b, op.output))
        elif isinstance(op, FloatSigmoidOp):
            operations.append(SigmoidOp(op.name, op.input, op.output))
        elif isinstance(op, FloatHardSigmoidOp):
            operations.append(HardSigmoidOp(op.name, op.input, op.output))
        elif isinstance(op, FloatHardSwishOp):
            operations.append(HardSwishOp(op.name, op.input, op.output))
        elif isinstance(op, FloatSiLUOp):
            operations.append(SiLUOp(op.name, op.input, op.output))
        elif isinstance(op, (FloatReLUOp, FloatReLU6Op)):
            output_qparams = qparams[op.output]
            clamp_input = requantized_input(op.input, output_qparams, op.name)
            maximum = 6.0 if isinstance(op, FloatReLU6Op) else math.inf
            activation_min = _quantize_code(0.0, output_qparams)
            activation_max = 127 if math.isinf(maximum) else _quantize_code(maximum, output_qparams)
            operations.append(
                ClampOp(op.name, clamp_input, op.output, activation_min, activation_max)
            )
        elif isinstance(op, (FloatAveragePool2DOp, FloatMaxPool2DOp)):
            input_type = values[op.input]
            if input_type.qparams == values[op.output].qparams:
                pool_output = op.output
            else:
                pool_output = unique_value(f"{op.output}.pool_domain")
                values[pool_output] = TensorType(
                    values[op.output].shape,
                    DType.INT8,
                    Layout.NHWC,
                    input_type.qparams,
                )
            pool_type = AveragePool2DOp if isinstance(op, FloatAveragePool2DOp) else MaxPool2DOp
            operations.append(
                pool_type(op.name, op.input, pool_output, op.kernel, op.stride, op.padding)
            )
            if pool_output != op.output:
                operations.append(
                    RequantizeOp(f"{op.name}.output_requantize", pool_output, op.output)
                )
        elif isinstance(op, (FloatAveragePool1DOp, FloatMaxPool1DOp)):
            input_type = values[op.input]
            if input_type.qparams == values[op.output].qparams:
                pool_output = op.output
            else:
                pool_output = unique_value(f"{op.output}.pool_domain")
                values[pool_output] = TensorType(
                    values[op.output].shape,
                    DType.INT8,
                    Layout.NLC,
                    input_type.qparams,
                )
            pool_type = AveragePool1DOp if isinstance(op, FloatAveragePool1DOp) else MaxPool1DOp
            operations.append(
                pool_type(op.name, op.input, pool_output, op.kernel, op.stride, op.padding)
            )
            if pool_output != op.output:
                operations.append(RequantizeOp(f"{op.name}.output_requantize", pool_output, op.output))
        elif isinstance(op, FloatPad2DOp):
            input_type = values[op.input]
            if input_type.qparams == values[op.output].qparams:
                pad_output = op.output
            else:
                pad_output = unique_value(f"{op.output}.pad_domain")
                values[pad_output] = TensorType(
                    values[op.output].shape,
                    DType.INT8,
                    Layout.NHWC,
                    input_type.qparams,
                )
            operations.append(Pad2DOp(op.name, op.input, pad_output, op.padding))
            if pad_output != op.output:
                operations.append(RequantizeOp(f"{op.name}.output_requantize", pad_output, op.output))
        elif isinstance(op, FloatReduceMeanOp):
            input_layout = graph.values[op.input].layout
            axes = (1, 2) if input_layout is FloatLayout.NCHW else (1,)
            operations.append(ReduceMeanOp(op.name, op.input, op.output, axes, True))
        elif isinstance(op, FloatResizeNearest2DOp):
            operations.append(ResizeNearest2DOp(op.name, op.input, op.output))
        elif isinstance(op, FloatResizeBilinear2DOp):
            operations.append(
                ResizeBilinear2DOp(op.name, op.input, op.output, op.align_corners)
            )
        elif isinstance(op, FloatSliceOp):
            input_layout = graph.values[op.input].layout
            if input_layout is FloatLayout.NCHW:
                axis_map = (0, 3, 1, 2)
            elif input_layout is FloatLayout.NCL:
                axis_map = (0, 2, 1)
            else:
                axis_map = tuple(range(len(graph.values[op.input].shape)))
            operations.append(
                SliceOp(
                    op.name,
                    op.input,
                    op.output,
                    axis_map[op.axis],
                    op.start,
                    op.stop,
                    op.step,
                )
            )
        elif isinstance(op, (FloatFlattenOp, FloatReshapeOp)):
            input_float = graph.values[op.input]
            output_float = graph.values[op.output]
            if input_float.layout is FloatLayout.NCHW and output_float.layout is FloatLayout.NC:
                consumers = consumer_map.get(op.output, ())
                if len(consumers) != 1 or not isinstance(consumers[0], FloatLinearOp):
                    raise CompileError(
                        f"{op.name}: NCHW-to-NC view is supported only when consumed by one Linear; "
                        "canonical NHWC flatten order otherwise changes semantics"
                    )
                _, channels, height, width = input_float.shape
                flattened_permutations[op.output] = np.arange(
                    channels * height * width, dtype=np.int64
                ).reshape(channels, height, width).transpose(1, 2, 0).reshape(-1)
            elif input_float.layout is FloatLayout.NCHW:
                if output_float.layout is FloatLayout.NCL:
                    if 1 not in input_float.shape[2:]:
                        raise CompileError(
                            f"{op.name}: NCHW-to-NCL Squeeze requires one singleton spatial axis"
                        )
                elif output_float.layout is not FloatLayout.NCHW or input_float.shape != output_float.shape:
                    raise CompileError(
                        f"{op.name}: general NCHW Reshape is not layout-preserving after NHWC canonicalization"
                    )
            elif input_float.layout is FloatLayout.NCL:
                if output_float.layout is FloatLayout.NCHW:
                    if 1 not in output_float.shape[2:]:
                        raise CompileError(
                            f"{op.name}: NCL-to-NCHW Unsqueeze requires one singleton spatial axis"
                        )
                elif output_float.layout is FloatLayout.NC:
                    if input_float.shape[2] != 1:
                        raise CompileError(f"{op.name}: NCL-to-NC Squeeze requires length one")
                elif output_float.layout is not FloatLayout.NCL or input_float.shape != output_float.shape:
                    raise CompileError(
                        f"{op.name}: general NCL Reshape changes NLC canonical order"
                    )
            elif input_float.layout is FloatLayout.NC:
                if output_float.layout is FloatLayout.NCL:
                    if output_float.shape[2] != 1:
                        raise CompileError(f"{op.name}: NC-to-NCL Unsqueeze requires length one")
                elif output_float.layout is not FloatLayout.NC:
                    raise CompileError(f"{op.name}: unsupported NC view transition")
            else:
                raise CompileError(f"{op.name}: unsupported view layout transition")
            view_input = requantized_input(op.input, qparams[op.output], op.name)
            view_type = FlattenOp if isinstance(op, FloatFlattenOp) else ReshapeOp
            operations.append(view_type(op.name, view_input, op.output))
        elif isinstance(op, FloatConcatOp):
            input_layout = graph.values[op.input_values[0]].layout
            axis = op.axis
            if input_layout is FloatLayout.NCHW:
                axis = {0: 0, 1: 3, 2: 1, 3: 2}[axis]
            elif input_layout is FloatLayout.NCL:
                axis = {0: 0, 1: 2, 2: 1}[axis]
            operations.append(ConcatenateOp(op.name, op.input_values, op.output, axis))
        elif isinstance(op, FloatSoftmaxOp):
            operations.append(SoftmaxOp(op.name, op.input, op.output))
        else:
            raise CompileError(f"{op.name}: PTQ lowering does not support {type(op).__name__}")

    raw = QuantizedGraph(
        name=graph.name if name is None else name,
        values=values,
        constants=constants,
        ops=tuple(operations),
        inputs=graph.inputs,
        outputs=graph.outputs,
        arithmetic_profile=ARITHMETIC_PROFILE,
    )
    legalized = legalize_graph(raw)
    fused = fuse_clamps(legalized)
    verify_graph(fused)
    return fused


__all__ = ["quantize_float_graph"]
