from __future__ import annotations

import numpy as np

from bakenn.ir.types import PerTensorQParams
from bakenn.plan.steps.spatial import ConvTranspose2DStep, ResizeBilinear2DStep, ResizeNearest2DStep

from ..contracts import ConstantEmission, KernelEmission, StepEmitContext, StepEmission, emit_step
from ..fixedpoint import clamp_s8_name, q31_kernel, q31_requantize_name
from ..formatting import format_values


def _uint32_constant(symbol: str, values: tuple[int, ...]) -> ConstantEmission:
    array = np.asarray(values, dtype=np.uint32)
    return ConstantEmission(
        symbol,
        f"extern const uint32_t {symbol}[{array.size}];",
        f"const uint32_t {symbol}[{array.size}] = {{\n{format_values(array)}\n}};",
        int(array.nbytes),
        alignment=4,
    )


def _int32_constant(symbol: str, values: tuple[int, ...]) -> ConstantEmission:
    array = np.asarray(values, dtype=np.int32)
    return ConstantEmission(
        symbol,
        f"extern const int32_t {symbol}[{array.size}];",
        f"const int32_t {symbol}[{array.size}] = {{\n{format_values(array)}\n}};",
        int(array.nbytes),
        alignment=4,
    )


def _resize_kernel(context: StepEmitContext) -> KernelEmission:
    symbol = context.symbol
    return KernelEmission(
        key="resize2d_s8_v1",
        header_includes=("<stddef.h>", "<stdint.h>"),
        declaration=f"""void {symbol}_resize_nearest2d_s8(
    const int8_t *input, int8_t *output,
    size_t input_width,
    size_t output_height, size_t output_width, size_t channels,
    const uint32_t *y_indices, const uint32_t *x_indices);

void {symbol}_resize_bilinear2d_s8(
    const int8_t *input, int8_t *output,
    size_t input_width, size_t output_height, size_t output_width, size_t channels,
    const uint32_t *y0, const uint32_t *y1, const uint32_t *yw_q15,
    const uint32_t *x0, const uint32_t *x1, const uint32_t *xw_q15);""",
        definition=f"""static int32_t {symbol}_round_q30_away(int64_t value) {{
    const uint64_t magnitude = value < 0 ? (uint64_t)(-(value + INT64_C(1))) + UINT64_C(1) : (uint64_t)value;
    uint64_t quotient = magnitude / (UINT64_C(1) << 30);
    const uint64_t remainder = magnitude % (UINT64_C(1) << 30);
    if (remainder >= (UINT64_C(1) << 29)) quotient += UINT64_C(1);
    return value < 0 ? -(int32_t)quotient : (int32_t)quotient;
}}

void {symbol}_resize_nearest2d_s8(
    const int8_t *input, int8_t *output,
    size_t input_width,
    size_t output_height, size_t output_width, size_t channels,
    const uint32_t *y_indices, const uint32_t *x_indices) {{
    for (size_t y = 0; y < output_height; ++y) {{
        const size_t source_y = y_indices[y];
        for (size_t x = 0; x < output_width; ++x) {{
            const size_t source_x = x_indices[x];
            for (size_t channel = 0; channel < channels; ++channel) {{
                output[(y * output_width + x) * channels + channel] =
                    input[(source_y * input_width + source_x) * channels + channel];
            }}
        }}
    }}
}}

void {symbol}_resize_bilinear2d_s8(
    const int8_t *input, int8_t *output,
    size_t input_width, size_t output_height, size_t output_width, size_t channels,
    const uint32_t *y0, const uint32_t *y1, const uint32_t *yw_q15,
    const uint32_t *x0, const uint32_t *x1, const uint32_t *xw_q15) {{
    for (size_t y = 0; y < output_height; ++y) {{
        const int64_t wy = (int64_t)yw_q15[y];
        for (size_t x = 0; x < output_width; ++x) {{
            const int64_t wx = (int64_t)xw_q15[x];
            const int64_t w00 = (INT64_C(32768) - wy) * (INT64_C(32768) - wx);
            const int64_t w01 = (INT64_C(32768) - wy) * wx;
            const int64_t w10 = wy * (INT64_C(32768) - wx);
            const int64_t w11 = wy * wx;
            for (size_t channel = 0; channel < channels; ++channel) {{
                const int64_t numerator =
                    (int64_t)input[((size_t)y0[y] * input_width + x0[x]) * channels + channel] * w00 +
                    (int64_t)input[((size_t)y0[y] * input_width + x1[x]) * channels + channel] * w01 +
                    (int64_t)input[((size_t)y1[y] * input_width + x0[x]) * channels + channel] * w10 +
                    (int64_t)input[((size_t)y1[y] * input_width + x1[x]) * channels + channel] * w11;
                int32_t result = {symbol}_round_q30_away(numerator);
                if (result < -128) result = -128;
                if (result > 127) result = 127;
                output[(y * output_width + x) * channels + channel] = (int8_t)result;
            }}
        }}
    }}
}}""",
    )


def _transpose_kernel(context: StepEmitContext) -> KernelEmission:
    symbol = context.symbol
    requantize = q31_requantize_name(context)
    clamp = clamp_s8_name(context)
    return KernelEmission(
        key="conv_transpose2d_s8_v1",
        header_includes=("<stddef.h>", "<stdint.h>"),
        declaration=f"""void {symbol}_conv_transpose2d_s8(
    const int8_t *input, const int8_t *weight, const int32_t *bias,
    const int32_t *multipliers, const int32_t *shifts, int8_t *output,
    size_t input_height, size_t input_width, size_t input_channels,
    size_t output_height, size_t output_width, size_t output_channels,
    size_t groups,
    size_t kernel_height, size_t kernel_width,
    size_t stride_height, size_t stride_width,
    size_t dilation_height, size_t dilation_width,
    size_t pad_top, size_t pad_left,
    int32_t input_zero_point, int32_t output_zero_point,
    int32_t activation_min, int32_t activation_max);""",
        definition=f"""void {symbol}_conv_transpose2d_s8(
    const int8_t *input, const int8_t *weight, const int32_t *bias,
    const int32_t *multipliers, const int32_t *shifts, int8_t *output,
    size_t input_height, size_t input_width, size_t input_channels,
    size_t output_height, size_t output_width, size_t output_channels,
    size_t groups,
    size_t kernel_height, size_t kernel_width,
    size_t stride_height, size_t stride_width,
    size_t dilation_height, size_t dilation_width,
    size_t pad_top, size_t pad_left,
    int32_t input_zero_point, int32_t output_zero_point,
    int32_t activation_min, int32_t activation_max) {{
    const size_t input_channels_per_group = input_channels / groups;
    const size_t output_channels_per_group = output_channels / groups;
    for (size_t output_y = 0; output_y < output_height; ++output_y) {{
        for (size_t output_x = 0; output_x < output_width; ++output_x) {{
            for (size_t output_channel = 0; output_channel < output_channels; ++output_channel) {{
                int32_t accumulator = bias[output_channel];
                const size_t group = output_channel / output_channels_per_group;
                const size_t input_channel_start = group * input_channels_per_group;
                for (size_t kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {{
                    const int64_t candidate_y = (int64_t)output_y + (int64_t)pad_top -
                        (int64_t)kernel_y * (int64_t)dilation_height;
                    if (candidate_y < 0 || (uint64_t)candidate_y % stride_height != 0u) continue;
                    const size_t input_y = (size_t)candidate_y / stride_height;
                    if (input_y >= input_height) continue;
                    for (size_t kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {{
                        const int64_t candidate_x = (int64_t)output_x + (int64_t)pad_left -
                            (int64_t)kernel_x * (int64_t)dilation_width;
                        if (candidate_x < 0 || (uint64_t)candidate_x % stride_width != 0u) continue;
                        const size_t input_x = (size_t)candidate_x / stride_width;
                        if (input_x >= input_width) continue;
                        for (size_t local_input_channel = 0;
                             local_input_channel < input_channels_per_group;
                             ++local_input_channel) {{
                            const size_t input_channel = input_channel_start + local_input_channel;
                            const size_t input_index =
                                (input_y * input_width + input_x) * input_channels + input_channel;
                            const size_t weight_index =
                                (((output_channel * kernel_height + kernel_y) * kernel_width + kernel_x) *
                                 input_channels_per_group) + local_input_channel;
                            const int32_t centered = (int32_t)input[input_index] - input_zero_point;
                            accumulator += centered * (int32_t)weight[weight_index];
                        }}
                    }}
                }}
                const int32_t scaled = {requantize}(
                    accumulator, multipliers[output_channel], shifts[output_channel]);
                output[(output_y * output_width + output_x) * output_channels + output_channel] =
                    {clamp}((int64_t)scaled + output_zero_point, activation_min, activation_max);
            }}
        }}
    }}
}}""",
    )


@emit_step.register
def _emit_nearest(step: ResizeNearest2DStep, context: StepEmitContext) -> StepEmission:
    prefix = f"{context.symbol}_op{context.step_index}"
    y_name = f"{prefix}_y"
    x_name = f"{prefix}_x"
    constants = (
        _uint32_constant(y_name, step.y_indices),
        _uint32_constant(x_name, step.x_indices),
    )
    input_shape = context.plan.tensors[step.input].tensor_type.shape
    output_shape = context.plan.tensors[step.output].tensor_type.shape
    return StepEmission(
        constants, (_resize_kernel(context),),
        f"    {context.symbol}_resize_nearest2d_s8({context.pointer(step.input, mutable=False)}, "
        f"{context.pointer(step.output, mutable=True)}, {input_shape[2]}u, "
        f"{output_shape[1]}u, {output_shape[2]}u, {output_shape[3]}u, "
        f"{y_name}, {x_name});",
        {"name": step.name, "kind": step.kernel_kind, "input_shape": list(input_shape), "output_shape": list(output_shape), "map_bytes": sum(item.size_bytes for item in constants)},
    )


@emit_step.register
def _emit_bilinear(step: ResizeBilinear2DStep, context: StepEmitContext) -> StepEmission:
    prefix = f"{context.symbol}_op{context.step_index}"
    names = [f"{prefix}_{suffix}" for suffix in ("y0", "y1", "yw", "x0", "x1", "xw")]
    constants = tuple(
        _uint32_constant(name, values)
        for name, values in zip(names, (step.y0, step.y1, step.yw_q15, step.x0, step.x1, step.xw_q15))
    )
    input_shape = context.plan.tensors[step.input].tensor_type.shape
    output_shape = context.plan.tensors[step.output].tensor_type.shape
    return StepEmission(
        constants, (_resize_kernel(context),),
        f"    {context.symbol}_resize_bilinear2d_s8({context.pointer(step.input, mutable=False)}, "
        f"{context.pointer(step.output, mutable=True)}, {input_shape[2]}u, {output_shape[1]}u, "
        f"{output_shape[2]}u, {output_shape[3]}u, {', '.join(names)});",
        {"name": step.name, "kind": step.kernel_kind, "align_corners": step.align_corners, "map_bytes": sum(item.size_bytes for item in constants)},
    )


@emit_step.register
def _emit_transpose(step: ConvTranspose2DStep, context: StepEmitContext) -> StepEmission:
    multiplier = f"{context.symbol}_op{context.step_index}_multiplier"
    shift = f"{context.symbol}_op{context.step_index}_shift"
    constants = (_int32_constant(multiplier, step.multipliers), _int32_constant(shift, step.shifts))
    input_type = context.plan.tensors[step.input].tensor_type
    output_type = context.plan.tensors[step.output].tensor_type
    weight_type = context.plan.tensors[step.weight].tensor_type
    input_qparams = input_type.qparams
    output_qparams = output_type.qparams
    assert isinstance(input_qparams, PerTensorQParams)
    assert isinstance(output_qparams, PerTensorQParams)
    return StepEmission(
        constants, (q31_kernel(context), _transpose_kernel(context)),
        f"""    {context.symbol}_conv_transpose2d_s8(
        {context.pointer(step.input, mutable=False)}, {context.pointer(step.weight, mutable=False)},
        {context.pointer(step.bias, mutable=False)}, {multiplier}, {shift},
        {context.pointer(step.output, mutable=True)},
        {input_type.shape[1]}u, {input_type.shape[2]}u, {input_type.shape[3]}u,
        {output_type.shape[1]}u, {output_type.shape[2]}u, {output_type.shape[3]}u,
        {step.groups}u,
        {weight_type.shape[1]}u, {weight_type.shape[2]}u,
        {step.stride[0]}u, {step.stride[1]}u, {step.dilation[0]}u, {step.dilation[1]}u,
        {step.padding[0]}u, {step.padding[2]}u,
        {input_qparams.zero_point}, {output_qparams.zero_point},
        {step.activation_min}, {step.activation_max});""",
        {"name": step.name, "kind": step.kernel_kind, "groups": step.groups, "stride": list(step.stride), "dilation": list(step.dilation), "padding": list(step.padding), "output_padding": list(step.output_padding), "accumulator_bound_max": max(step.accumulator_bounds)},
    )


__all__: list[str] = []
