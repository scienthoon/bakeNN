from __future__ import annotations

import numpy as np

from bakenn.errors import CompileError
from bakenn.ir.graph import QuantizedGraph
from bakenn.ir.ops.spatial import ConvTranspose2DOp, ResizeBilinear2DOp, ResizeNearest2DOp
from bakenn.ir.types import PerAxisQParams, PerTensorQParams
from bakenn.plan.lower import lower_op
from bakenn.plan.steps.spatial import ConvTranspose2DStep, ResizeBilinear2DStep, ResizeNearest2DStep
from bakenn.quantization.fixedpoint import INT32_MAX, quantize_multiplier


def _axis_map(input_size: int, output_size: int, align_corners: bool) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    lower: list[int] = []
    upper: list[int] = []
    weights: list[int] = []
    for output_index in range(output_size):
        if align_corners and output_size > 1:
            numerator = output_index * (input_size - 1)
            denominator = output_size - 1
        else:
            numerator = (2 * output_index + 1) * input_size - output_size
            denominator = 2 * output_size
        if numerator <= 0:
            index0 = index1 = 0
            weight = 0
        elif numerator >= (input_size - 1) * denominator:
            index0 = index1 = input_size - 1
            weight = 0
        else:
            index0, remainder = divmod(numerator, denominator)
            index1 = index0 + 1
            weight = (remainder * 32768 + denominator // 2) // denominator
        lower.append(index0)
        upper.append(index1)
        weights.append(weight)
    return tuple(lower), tuple(upper), tuple(weights)


@lower_op.register
def _lower_nearest(op: ResizeNearest2DOp, graph: QuantizedGraph) -> ResizeNearest2DStep:
    input_shape = graph.values[op.input].shape
    output_shape = graph.values[op.output].shape
    y_indices = tuple(
        output_index * input_shape[1] // output_shape[1]
        for output_index in range(output_shape[1])
    )
    x_indices = tuple(
        output_index * input_shape[2] // output_shape[2]
        for output_index in range(output_shape[2])
    )
    return ResizeNearest2DStep(
        op.name,
        op.input,
        op.output,
        y_indices,
        x_indices,
    )


@lower_op.register
def _lower_bilinear(op: ResizeBilinear2DOp, graph: QuantizedGraph) -> ResizeBilinear2DStep:
    input_shape = graph.values[op.input].shape
    output_shape = graph.values[op.output].shape
    y0, y1, yw = _axis_map(input_shape[1], output_shape[1], op.align_corners)
    x0, x1, xw = _axis_map(input_shape[2], output_shape[2], op.align_corners)
    return ResizeBilinear2DStep(
        op.name, op.input, op.output, op.align_corners, y0, y1, yw, x0, x1, xw
    )


@lower_op.register
def _lower_transpose(op: ConvTranspose2DOp, graph: QuantizedGraph) -> ConvTranspose2DStep:
    input_type = graph.values[op.input]
    weight_type = graph.values[op.weight]
    output_type = graph.values[op.output]
    assert isinstance(input_type.qparams, PerTensorQParams)
    assert isinstance(weight_type.qparams, PerAxisQParams)
    assert isinstance(output_type.qparams, PerTensorQParams)
    weight = np.asarray(graph.constants[op.weight], dtype=np.int8)
    bias = np.asarray(graph.constants[op.bias], dtype=np.int32)
    output_channels = weight.shape[0]
    max_abs_input = max(abs(-128 - input_type.qparams.zero_point), abs(127 - input_type.qparams.zero_point))
    bounds: list[int] = []
    multipliers: list[int] = []
    shifts: list[int] = []
    for channel in range(output_channels):
        bound = abs(int(bias[channel])) + max_abs_input * sum(abs(int(value)) for value in weight[channel].flat)
        if bound > INT32_MAX:
            raise CompileError(f"{op.name} channel {channel}: accumulator bound {bound} exceeds int32")
        multiplier, shift = quantize_multiplier(
            input_type.qparams.scale * weight_type.qparams.scales[channel] / output_type.qparams.scale
        )
        if shift > 0 and bound * (1 << shift) > INT32_MAX:
            raise CompileError(f"{op.name} channel {channel}: requantization left shift is not int32-safe")
        bounds.append(bound)
        multipliers.append(multiplier)
        shifts.append(shift)
    return ConvTranspose2DStep(
        op.name, op.input, op.weight, op.bias, op.output,
        op.stride, op.dilation, op.padding, op.output_padding, op.groups,
        tuple(multipliers), tuple(shifts), op.activation_min, op.activation_max, tuple(bounds),
    )


__all__: list[str] = []
