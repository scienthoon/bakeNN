from __future__ import annotations

import numpy as np

from bakenn.errors import CompileError
from bakenn.ir.types import PerTensorQParams
from bakenn.plan import ExecutionPlan
from bakenn.plan.steps.conv import Conv2DStep, DepthwiseConv2DStep

from ..contracts import ConstantEmission, KernelEmission, StepEmitContext, StepEmission, emit_step
from ..fixedpoint import clamp_s8_name, q31_kernel, q31_requantize_name
from ..formatting import format_values
from ..selection import CBackendOptions, KernelCapability, PackedConstant, kernel_capabilities


_PORTABLE_CONV_ID = "portable.conv2d_s8.v1"
_CONV_1X1_ID = "optimized.conv2d_1x1_o2.v1"
_CORTEX_M4_CONV_1X1_ID = "cortex_m4.conv2d_1x1_smlad.v1"
_CORTEX_M4_CONV_3X3_ID = "cortex_m4.conv2d_3x3_im2col_smlad.v1"
_PORTABLE_DEPTHWISE_ID = "portable.depthwise_conv2d_s8.v1"
_DEPTHWISE_3X3_ID = "optimized.depthwise_3x3_c2.v1"
_CORTEX_M4_DEPTHWISE_ID = "cortex_m4.depthwise_3x3_smlad.v1"


def _pack_signed_i8_pairs(rows: np.ndarray) -> np.ndarray:
    """Pack each row as two sign-extended int16 lanes in one int32 word."""

    matrix = np.ascontiguousarray(rows.reshape(rows.shape[0], -1), dtype=np.int8)
    pair_count = (matrix.shape[1] + 1) // 2
    packed = np.zeros((matrix.shape[0], pair_count), dtype=np.int32)
    for row in range(matrix.shape[0]):
        for pair in range(pair_count):
            low = int(matrix[row, pair * 2]) & 0xFFFF
            high_index = pair * 2 + 1
            high = int(matrix[row, high_index]) & 0xFFFF if high_index < matrix.shape[1] else 0
            packed[row, pair] = np.asarray(low | (high << 16), dtype=np.uint32).view(np.int32)
    return packed


def _int32_constant(symbol: str, values: tuple[int, ...]) -> ConstantEmission:
    array = np.asarray(values, dtype=np.int32)
    return ConstantEmission(
        symbol=symbol,
        declaration=f"extern const int32_t {symbol}[{len(values)}];",
        definition=f"const int32_t {symbol}[{len(values)}] = {{\n{format_values(array)}\n}};",
        size_bytes=int(array.nbytes),
    )


def _conv2d_kernel(context: StepEmitContext) -> KernelEmission:
    symbol = context.symbol
    kernel_fn = f"{symbol}_conv2d_s8"
    requantize = q31_requantize_name(context)
    clamp = clamp_s8_name(context)
    return KernelEmission(
        key="conv2d_s8_v1",
        header_includes=("<stddef.h>", "<stdint.h>"),
        declaration=f"""void {kernel_fn}(
    const int8_t *input,
    const int8_t *weight,
    const int32_t *bias,
    const int32_t *multiplier,
    const int32_t *shift,
    int8_t *output,
    size_t input_height,
    size_t input_width,
    size_t input_channels,
    size_t output_height,
    size_t output_width,
    size_t output_channels,
    size_t groups,
    size_t kernel_height,
    size_t kernel_width,
    size_t stride_height,
    size_t stride_width,
    size_t dilation_height,
    size_t dilation_width,
    size_t pad_top,
    size_t pad_left,
    int32_t input_zero_point,
    int32_t output_zero_point,
    int32_t activation_min,
    int32_t activation_max);""",
        definition=f"""void {kernel_fn}(
    const int8_t *input,
    const int8_t *weight,
    const int32_t *bias,
    const int32_t *multiplier,
    const int32_t *shift,
    int8_t *output,
    size_t input_height,
    size_t input_width,
    size_t input_channels,
    size_t output_height,
    size_t output_width,
    size_t output_channels,
    size_t groups,
    size_t kernel_height,
    size_t kernel_width,
    size_t stride_height,
    size_t stride_width,
    size_t dilation_height,
    size_t dilation_width,
    size_t pad_top,
    size_t pad_left,
    int32_t input_zero_point,
    int32_t output_zero_point,
    int32_t activation_min,
    int32_t activation_max) {{
    for (size_t output_y = 0; output_y < output_height; ++output_y) {{
        for (size_t output_x = 0; output_x < output_width; ++output_x) {{
            const size_t group_input_channels = input_channels / groups;
            const size_t group_output_channels = output_channels / groups;
            for (size_t output_channel = 0; output_channel < output_channels; ++output_channel) {{
                const size_t group = output_channel / group_output_channels;
                const size_t input_channel_base = group * group_input_channels;
                int32_t accumulator = bias[output_channel];
                for (size_t kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {{
                    const int64_t input_y = (int64_t)output_y * (int64_t)stride_height
                        + (int64_t)kernel_y * (int64_t)dilation_height - (int64_t)pad_top;
                    for (size_t kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {{
                        const int64_t input_x = (int64_t)output_x * (int64_t)stride_width
                            + (int64_t)kernel_x * (int64_t)dilation_width - (int64_t)pad_left;
                        for (size_t local_input_channel = 0;
                             local_input_channel < group_input_channels;
                             ++local_input_channel) {{
                            const size_t input_channel =
                                input_channel_base + local_input_channel;
                            int32_t input_value = input_zero_point;
                            if (input_y >= 0 && input_x >= 0
                                && (uint64_t)input_y < (uint64_t)input_height
                                && (uint64_t)input_x < (uint64_t)input_width) {{
                                const size_t input_index =
                                    ((size_t)input_y * input_width + (size_t)input_x) * input_channels
                                    + input_channel;
                                input_value = input[input_index];
                            }}
                            const size_t weight_index =
                                ((output_channel * kernel_height + kernel_y) * kernel_width + kernel_x)
                                * group_input_channels + local_input_channel;
                            accumulator += (input_value - input_zero_point) * (int32_t)weight[weight_index];
                        }}
                    }}
                }}
                const int32_t scaled = {requantize}(
                    accumulator, multiplier[output_channel], shift[output_channel]);
                const size_t output_index =
                    (output_y * output_width + output_x) * output_channels + output_channel;
                output[output_index] = {clamp}(
                    (int64_t)scaled + output_zero_point, activation_min, activation_max);
            }}
        }}
    }}
}}""",
    )


def _conv1x1_o2_kernel(context: StepEmitContext) -> KernelEmission:
    symbol = context.symbol
    kernel_fn = f"{symbol}_conv2d_1x1_o2_s8"
    requantize = q31_requantize_name(context)
    clamp = clamp_s8_name(context)
    signature = f"""void {kernel_fn}(
    const int8_t *input,
    const int8_t *weight,
    const int32_t *bias,
    const int32_t *multiplier,
    const int32_t *shift,
    int8_t *output,
    size_t input_height,
    size_t input_width,
    size_t input_channels,
    size_t output_height,
    size_t output_width,
    size_t output_channels,
    size_t stride_height,
    size_t stride_width,
    int32_t input_zero_point,
    int32_t output_zero_point,
    int32_t activation_min,
    int32_t activation_max)"""
    return KernelEmission(
        key="conv2d_1x1_o2_s8_v1",
        header_includes=("<stddef.h>", "<stdint.h>"),
        declaration=signature + ";",
definition=f"""{signature} {{
    (void)input_height;
    for (size_t output_y = 0; output_y < output_height; ++output_y) {{
        const size_t input_y = output_y * stride_height;
        for (size_t output_x = 0; output_x < output_width; ++output_x) {{
            const size_t input_x = output_x * stride_width;
            const int8_t *input_pixel =
                input + (input_y * input_width + input_x) * input_channels;
            int8_t *output_pixel =
                output + (output_y * output_width + output_x) * output_channels;
            for (size_t output_channel = 0;
                 output_channel < output_channels;
                 output_channel += 2u) {{
                int32_t accumulator_0 = bias[output_channel];
                int32_t accumulator_1 = bias[output_channel + 1u];
                const int8_t *pair_weight =
                    weight + (output_channel / 2u) * input_channels * 2u;
                for (size_t input_channel = 0;
                     input_channel < input_channels;
                     ++input_channel) {{
                    const int32_t centered =
                        (int32_t)input_pixel[input_channel] - input_zero_point;
                    const int8_t *pair = pair_weight + input_channel * 2u;
                    accumulator_0 += centered * (int32_t)pair[0];
                    accumulator_1 += centered * (int32_t)pair[1];
                }}
                const int32_t scaled_0 = {requantize}(
                    accumulator_0, multiplier[output_channel], shift[output_channel]);
                const int32_t scaled_1 = {requantize}(
                    accumulator_1, multiplier[output_channel + 1u],
                    shift[output_channel + 1u]);
                output_pixel[output_channel] = {clamp}(
                    (int64_t)scaled_0 + output_zero_point,
                    activation_min, activation_max);
                output_pixel[output_channel + 1u] = {clamp}(
                    (int64_t)scaled_1 + output_zero_point,
                    activation_min, activation_max);
            }}
        }}
    }}
}}""",
    )


def _cortex_m4_conv1x1_kernel(context: StepEmitContext) -> KernelEmission:
    symbol = context.symbol
    kernel_fn = f"{symbol}_conv2d_1x1_cortex_m4_smlad_s8"
    requantize = q31_requantize_name(context)
    clamp = clamp_s8_name(context)
    signature = f"""void {kernel_fn}(
    const int8_t *input,
    const int32_t *packed_weight,
    const int32_t *bias,
    const int32_t *multiplier,
    const int32_t *shift,
    int8_t *output,
    size_t input_width,
    size_t input_channels,
    size_t output_height,
    size_t output_width,
    size_t output_channels,
    size_t stride_height,
    size_t stride_width,
    int32_t input_zero_point,
    int32_t output_zero_point,
    int32_t activation_min,
    int32_t activation_max)"""
    return KernelEmission(
        key="cortex_m4_conv2d_1x1_smlad_s8_v1",
        header_includes=("<stddef.h>", "<stdint.h>"),
        declaration=signature + ";",
        definition=f"""{signature} {{
    const size_t pair_count = (input_channels + 1u) / 2u;
    for (size_t output_y = 0; output_y < output_height; ++output_y) {{
        for (size_t output_x = 0; output_x < output_width; ++output_x) {{
            const int8_t *input_pixel = input
                + (output_y * stride_height * input_width
                   + output_x * stride_width) * input_channels;
            int8_t *output_pixel = output
                + (output_y * output_width + output_x) * output_channels;
            for (size_t output_channel = 0;
                 output_channel < output_channels;
                 ++output_channel) {{
                int32_t accumulator = bias[output_channel];
                const int32_t *channel_weight =
                    packed_weight + output_channel * pair_count;
                for (size_t pair = 0; pair < pair_count; ++pair) {{
                    const size_t index = pair * 2u;
                    const int32_t low =
                        (int32_t)input_pixel[index] - input_zero_point;
                    const int32_t high = index + 1u < input_channels
                        ? (int32_t)input_pixel[index + 1u] - input_zero_point
                        : 0;
                    const uint32_t input_pair =
                        ((uint32_t)low & UINT32_C(0xffff))
                        | (((uint32_t)high & UINT32_C(0xffff)) << 16u);
                    accumulator = __builtin_arm_smlad(
                        (int32_t)input_pair, channel_weight[pair], accumulator);
                }}
                const int32_t scaled = {requantize}(
                    accumulator, multiplier[output_channel], shift[output_channel]);
                output_pixel[output_channel] = {clamp}(
                    (int64_t)scaled + output_zero_point,
                    activation_min, activation_max);
            }}
        }}
    }}
}}""",
    )


def _cortex_m4_conv3x3_kernel(context: StepEmitContext) -> KernelEmission:
    symbol = context.symbol
    kernel_fn = f"{symbol}_conv2d_3x3_cortex_m4_smlad_s8"
    requantize = q31_requantize_name(context)
    clamp = clamp_s8_name(context)
    signature = f"""void {kernel_fn}(
    const int8_t *input,
    const int32_t *packed_weight,
    const int32_t *bias,
    const int32_t *multiplier,
    const int32_t *shift,
    int8_t *output,
    int32_t *patch,
    size_t input_height,
    size_t input_width,
    size_t input_channels,
    size_t output_height,
    size_t output_width,
    size_t output_channels,
    size_t pad_top,
    size_t pad_left,
    int32_t input_zero_point,
    int32_t output_zero_point,
    int32_t activation_min,
    int32_t activation_max)"""
    return KernelEmission(
        key="cortex_m4_conv2d_3x3_im2col_smlad_s8_v1",
        header_includes=("<stddef.h>", "<stdint.h>"),
        declaration=signature + ";",
        definition=f"""{signature} {{
    const size_t mac_count = 9u * input_channels;
    const size_t pair_count = (mac_count + 1u) / 2u;
    for (size_t output_y = 0; output_y < output_height; ++output_y) {{
        for (size_t output_x = 0; output_x < output_width; ++output_x) {{
            for (size_t pair = 0; pair < pair_count; ++pair) {{
                uint32_t lanes = 0u;
                for (size_t lane = 0; lane < 2u; ++lane) {{
                    const size_t mac = pair * 2u + lane;
                    int32_t centered = 0;
                    if (mac < mac_count) {{
                        const size_t kernel_position = mac / input_channels;
                        const size_t input_channel = mac % input_channels;
                        const int64_t input_y = (int64_t)output_y
                            + (int64_t)(kernel_position / 3u) - (int64_t)pad_top;
                        const int64_t input_x = (int64_t)output_x
                            + (int64_t)(kernel_position % 3u) - (int64_t)pad_left;
                        if (input_y >= 0 && input_x >= 0
                            && (uint64_t)input_y < (uint64_t)input_height
                            && (uint64_t)input_x < (uint64_t)input_width) {{
                            const size_t input_index =
                                ((size_t)input_y * input_width + (size_t)input_x)
                                * input_channels + input_channel;
                            centered = (int32_t)input[input_index] - input_zero_point;
                        }}
                    }}
                    lanes |= ((uint32_t)centered & UINT32_C(0xffff))
                        << (lane * 16u);
                }}
                patch[pair] = (int32_t)lanes;
            }}
            int8_t *output_pixel = output
                + (output_y * output_width + output_x) * output_channels;
            for (size_t output_channel = 0;
                 output_channel < output_channels;
                 ++output_channel) {{
                int32_t accumulator = bias[output_channel];
                const int32_t *channel_weight =
                    packed_weight + output_channel * pair_count;
                for (size_t pair = 0; pair < pair_count; ++pair) {{
                    accumulator = __builtin_arm_smlad(
                        patch[pair], channel_weight[pair], accumulator);
                }}
                const int32_t scaled = {requantize}(
                    accumulator, multiplier[output_channel], shift[output_channel]);
                output_pixel[output_channel] = {clamp}(
                    (int64_t)scaled + output_zero_point,
                    activation_min, activation_max);
            }}
        }}
    }}
}}""",
    )


def _depthwise_kernel(context: StepEmitContext) -> KernelEmission:
    symbol = context.symbol
    kernel_fn = f"{symbol}_depthwise_conv2d_s8"
    requantize = q31_requantize_name(context)
    clamp = clamp_s8_name(context)
    return KernelEmission(
        key="depthwise_conv2d_s8_v1",
        header_includes=("<stddef.h>", "<stdint.h>"),
        declaration=f"""void {kernel_fn}(
    const int8_t *input,
    const int8_t *weight,
    const int32_t *bias,
    const int32_t *multiplier,
    const int32_t *shift,
    int8_t *output,
    size_t input_height,
    size_t input_width,
    size_t input_channels,
    size_t output_height,
    size_t output_width,
    size_t output_channels,
    size_t kernel_height,
    size_t kernel_width,
    size_t depth_multiplier,
    size_t stride_height,
    size_t stride_width,
    size_t dilation_height,
    size_t dilation_width,
    size_t pad_top,
    size_t pad_left,
    int32_t input_zero_point,
    int32_t output_zero_point,
    int32_t activation_min,
    int32_t activation_max);""",
        definition=f"""void {kernel_fn}(
    const int8_t *input,
    const int8_t *weight,
    const int32_t *bias,
    const int32_t *multiplier,
    const int32_t *shift,
    int8_t *output,
    size_t input_height,
    size_t input_width,
    size_t input_channels,
    size_t output_height,
    size_t output_width,
    size_t output_channels,
    size_t kernel_height,
    size_t kernel_width,
    size_t depth_multiplier,
    size_t stride_height,
    size_t stride_width,
    size_t dilation_height,
    size_t dilation_width,
    size_t pad_top,
    size_t pad_left,
    int32_t input_zero_point,
    int32_t output_zero_point,
    int32_t activation_min,
    int32_t activation_max) {{
    (void)input_channels;
    for (size_t output_y = 0; output_y < output_height; ++output_y) {{
        for (size_t output_x = 0; output_x < output_width; ++output_x) {{
            for (size_t output_channel = 0; output_channel < output_channels; ++output_channel) {{
                const size_t input_channel = output_channel / depth_multiplier;
                int32_t accumulator = bias[output_channel];
                for (size_t kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {{
                    const int64_t input_y = (int64_t)output_y * (int64_t)stride_height
                        + (int64_t)kernel_y * (int64_t)dilation_height - (int64_t)pad_top;
                    for (size_t kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {{
                        const int64_t input_x = (int64_t)output_x * (int64_t)stride_width
                            + (int64_t)kernel_x * (int64_t)dilation_width - (int64_t)pad_left;
                        int32_t input_value = input_zero_point;
                        if (input_y >= 0 && input_x >= 0
                            && (uint64_t)input_y < (uint64_t)input_height
                            && (uint64_t)input_x < (uint64_t)input_width) {{
                            const size_t input_index =
                                ((size_t)input_y * input_width + (size_t)input_x) * input_channels
                                + input_channel;
                            input_value = input[input_index];
                        }}
                        const size_t weight_index =
                            (kernel_y * kernel_width + kernel_x) * output_channels + output_channel;
                        accumulator +=
                            (input_value - input_zero_point) * (int32_t)weight[weight_index];
                    }}
                }}
                const int32_t scaled = {requantize}(
                    accumulator, multiplier[output_channel], shift[output_channel]);
                const size_t output_index =
                    (output_y * output_width + output_x) * output_channels + output_channel;
                output[output_index] = {clamp}(
                    (int64_t)scaled + output_zero_point, activation_min, activation_max);
            }}
        }}
    }}
}}""",
    )


def _depthwise_3x3_c2_kernel(context: StepEmitContext) -> KernelEmission:
    symbol = context.symbol
    kernel_fn = f"{symbol}_depthwise_3x3_c2_s8"
    requantize = q31_requantize_name(context)
    clamp = clamp_s8_name(context)
    signature = f"""void {kernel_fn}(
    const int8_t *input,
    const int8_t *weight,
    const int32_t *bias,
    const int32_t *multiplier,
    const int32_t *shift,
    int8_t *output,
    size_t input_height,
    size_t input_width,
    size_t input_channels,
    size_t output_height,
    size_t output_width,
    size_t output_channels,
    size_t stride_height,
    size_t stride_width,
    size_t pad_top,
    size_t pad_left,
    int32_t input_zero_point,
    int32_t output_zero_point,
    int32_t activation_min,
    int32_t activation_max)"""
    return KernelEmission(
        key="depthwise_3x3_c2_s8_v1",
        header_includes=("<stddef.h>", "<stdint.h>"),
        declaration=signature + ";",
        definition=f"""{signature} {{
    for (size_t output_y = 0; output_y < output_height; ++output_y) {{
        for (size_t output_x = 0; output_x < output_width; ++output_x) {{
            int8_t *output_pixel =
                output + (output_y * output_width + output_x) * output_channels;
            for (size_t output_channel = 0;
                 output_channel < output_channels;
                 output_channel += 2u) {{
                int32_t accumulator_0 = bias[output_channel];
                int32_t accumulator_1 = bias[output_channel + 1u];
                for (size_t kernel_y = 0; kernel_y < 3u; ++kernel_y) {{
                    const int64_t input_y =
                        (int64_t)output_y * (int64_t)stride_height
                        + (int64_t)kernel_y - (int64_t)pad_top;
                    for (size_t kernel_x = 0; kernel_x < 3u; ++kernel_x) {{
                        const int64_t input_x =
                            (int64_t)output_x * (int64_t)stride_width
                            + (int64_t)kernel_x - (int64_t)pad_left;
                        int32_t input_0 = input_zero_point;
                        int32_t input_1 = input_zero_point;
                        if (input_y >= 0 && input_x >= 0
                            && (uint64_t)input_y < (uint64_t)input_height
                            && (uint64_t)input_x < (uint64_t)input_width) {{
                            const size_t input_index =
                                ((size_t)input_y * input_width + (size_t)input_x)
                                * input_channels + output_channel;
                            input_0 = (int32_t)input[input_index];
                            input_1 = (int32_t)input[input_index + 1u];
                        }}
                        const int8_t *pair =
                            weight + (kernel_y * 3u + kernel_x) * output_channels
                            + output_channel;
                        accumulator_0 += (input_0 - input_zero_point)
                            * (int32_t)pair[0];
                        accumulator_1 += (input_1 - input_zero_point)
                            * (int32_t)pair[1];
                    }}
                }}
                const int32_t scaled_0 = {requantize}(
                    accumulator_0, multiplier[output_channel], shift[output_channel]);
                const int32_t scaled_1 = {requantize}(
                    accumulator_1, multiplier[output_channel + 1u],
                    shift[output_channel + 1u]);
                output_pixel[output_channel] = {clamp}(
                    (int64_t)scaled_0 + output_zero_point,
                    activation_min, activation_max);
                output_pixel[output_channel + 1u] = {clamp}(
                    (int64_t)scaled_1 + output_zero_point,
                    activation_min, activation_max);
            }}
        }}
    }}
}}""",
    )


def _cortex_m4_depthwise_kernel(context: StepEmitContext) -> KernelEmission:
    symbol = context.symbol
    kernel_fn = f"{symbol}_depthwise_3x3_cortex_m4_smlad_s8"
    requantize = q31_requantize_name(context)
    clamp = clamp_s8_name(context)
    signature = f"""void {kernel_fn}(
    const int8_t *input,
    const int32_t *packed_weight,
    const int32_t *bias,
    const int32_t *multiplier,
    const int32_t *shift,
    int8_t *output,
    size_t input_height,
    size_t input_width,
    size_t channels,
    size_t output_height,
    size_t output_width,
    size_t stride_height,
    size_t stride_width,
    size_t pad_top,
    size_t pad_left,
    int32_t input_zero_point,
    int32_t output_zero_point,
    int32_t activation_min,
    int32_t activation_max)"""
    return KernelEmission(
        key="cortex_m4_depthwise_3x3_smlad_s8_v1",
        header_includes=("<stddef.h>", "<stdint.h>"),
        declaration=signature + ";",
        definition=f"""{signature} {{
    for (size_t output_y = 0; output_y < output_height; ++output_y) {{
        for (size_t output_x = 0; output_x < output_width; ++output_x) {{
            int8_t *output_pixel =
                output + (output_y * output_width + output_x) * channels;
            for (size_t channel = 0; channel < channels; ++channel) {{
                int32_t accumulator = bias[channel];
                const int32_t *channel_weight = packed_weight + channel * 5u;
                for (size_t pair = 0; pair < 5u; ++pair) {{
                    uint32_t input_pair = 0u;
                    for (size_t lane = 0; lane < 2u; ++lane) {{
                        const size_t kernel_position = pair * 2u + lane;
                        int32_t centered = 0;
                        if (kernel_position < 9u) {{
                            const int64_t input_y =
                                (int64_t)output_y * (int64_t)stride_height
                                + (int64_t)(kernel_position / 3u) - (int64_t)pad_top;
                            const int64_t input_x =
                                (int64_t)output_x * (int64_t)stride_width
                                + (int64_t)(kernel_position % 3u) - (int64_t)pad_left;
                            if (input_y >= 0 && input_x >= 0
                                && (uint64_t)input_y < (uint64_t)input_height
                                && (uint64_t)input_x < (uint64_t)input_width) {{
                                const size_t input_index =
                                    ((size_t)input_y * input_width + (size_t)input_x)
                                    * channels + channel;
                                centered =
                                    (int32_t)input[input_index] - input_zero_point;
                            }}
                        }}
                        input_pair |= ((uint32_t)centered & UINT32_C(0xffff))
                            << (lane * 16u);
                    }}
                    accumulator = __builtin_arm_smlad(
                        (int32_t)input_pair, channel_weight[pair], accumulator);
                }}
                const int32_t scaled = {requantize}(
                    accumulator, multiplier[channel], shift[channel]);
                output_pixel[channel] = {clamp}(
                    (int64_t)scaled + output_zero_point,
                    activation_min, activation_max);
            }}
        }}
    }}
}}""",
    )


def _unsupported_capability(
    kernel_id: str,
    reason: str,
    *,
    priority: int = 100,
) -> KernelCapability:
    return KernelCapability(
        kernel_id=kernel_id,
        priority=priority,
        optimized=True,
        supported=False,
        reason=reason,
    )


@kernel_capabilities.register
def _conv2d_capabilities(
    step: Conv2DStep,
    plan: ExecutionPlan,
    options: CBackendOptions,
) -> tuple[KernelCapability, ...]:
    portable = KernelCapability(
        kernel_id=_PORTABLE_CONV_ID,
        priority=0,
        optimized=False,
        supported=True,
        reason="generic NHWC/OHWI Conv2D kernel supports every verified Conv2DStep",
    )
    input_type = plan.tensors[step.input].tensor_type
    output_type = plan.tensors[step.output].tensor_type
    weight = plan.constants[step.weight]
    output_channels = output_type.shape[3]
    input_channels = input_type.shape[3]
    def generic_1x1() -> KernelCapability:
        failure: str | None = None
        if step.groups != 1:
            failure = "1x1 optimized Conv2D requires groups one"
        elif not options.enable_weight_packing:
            failure = "weight packing is disabled"
        elif weight.dtype != np.int8 or weight.shape != (
            output_channels,
            1,
            1,
            input_channels,
        ):
            failure = "Conv2D weight must be an OI 1x1 int8 matrix"
        elif output_channels < 2 or output_channels % 2:
            failure = "1x1 output channels must be an even output-pair shape"
        elif step.dilation != (1, 1):
            failure = "1x1 optimized Conv2D requires dilation (1, 1)"
        elif step.stride[0] not in (1, 2) or step.stride[1] not in (1, 2):
            failure = "1x1 optimized Conv2D supports only stride components one or two"
        elif step.padding != (0, 0, 0, 0):
            failure = "1x1 optimized Conv2D requires zero explicit padding"
        if failure is not None:
            return _unsupported_capability(_CONV_1X1_ID, failure)
        packed_name = f"{step.weight}.conv2d_1x1_o2"
        packed_value = np.ascontiguousarray(
            weight.reshape(output_channels // 2, 2, input_channels).transpose(0, 2, 1)
        )
        packed = PackedConstant(
            name=packed_name,
            source=step.weight,
            layout="conv2d_1x1_ohwi_o2_interleaved_v1",
            value=packed_value,
        )
        return KernelCapability(
            kernel_id=_CONV_1X1_ID,
            priority=100,
            optimized=True,
            supported=True,
            reason=(
                "1x1 OHWI weights packed by output pairs; one centered input value "
                "is reused across two output accumulators"
            ),
            packed_constants=(packed,),
            constant_overrides={step.weight: packed.name},
        )

    def cortex_m4_1x1() -> KernelCapability:
        failure: str | None = None
        if "dsp" not in options.target.features or "armv7e-m" not in options.target.features:
            failure = "target does not provide ARMv7E-M DSP instructions"
        elif step.groups != 1:
            failure = "Cortex-M4 1x1 Conv2D requires groups one"
        elif not options.enable_weight_packing:
            failure = "weight packing is disabled"
        elif weight.dtype != np.int8 or weight.shape != (
            output_channels,
            1,
            1,
            input_channels,
        ):
            failure = "Cortex-M4 1x1 kernel requires matching OHWI int8 weights"
        elif step.dilation != (1, 1) or step.padding != (0, 0, 0, 0):
            failure = "Cortex-M4 1x1 kernel requires dilation one and zero padding"
        elif step.stride[0] not in (1, 2) or step.stride[1] not in (1, 2):
            failure = "Cortex-M4 1x1 kernel supports stride components one or two"
        if failure is not None:
            return _unsupported_capability(_CORTEX_M4_CONV_1X1_ID, failure, priority=300)
        packed_name = f"{step.weight}.cortex_m4_smlad"
        packed = PackedConstant(
            name=packed_name,
            source=step.weight,
            layout="conv2d_1x1_arm_smlad_i16x2_v1",
            value=_pack_signed_i8_pairs(weight),
            alignment=4,
        )
        return KernelCapability(
            kernel_id=_CORTEX_M4_CONV_1X1_ID,
            priority=300,
            optimized=True,
            supported=True,
            reason="ARMv7E-M SMLAD performs two 1x1 channel MACs per instruction",
            packed_constants=(packed,),
            constant_overrides={step.weight: packed.name},
        )

    def cortex_m4_3x3() -> KernelCapability:
        failure: str | None = None
        if "dsp" not in options.target.features or "armv7e-m" not in options.target.features:
            failure = "target does not provide ARMv7E-M DSP instructions"
        elif step.groups != 1:
            failure = "Cortex-M4 3x3 kernel requires groups one"
        elif not options.enable_weight_packing:
            failure = "weight packing is disabled"
        elif weight.dtype != np.int8 or weight.shape != (
            output_channels,
            3,
            3,
            input_channels,
        ):
            failure = "Cortex-M4 3x3 kernel requires matching OHWI int8 weights"
        elif step.stride != (1, 1) or step.dilation != (1, 1):
            failure = "Cortex-M4 3x3 v1 requires stride and dilation one"
        if failure is not None:
            return _unsupported_capability(_CORTEX_M4_CONV_3X3_ID, failure, priority=290)
        packed_name = f"{step.weight}.cortex_m4_3x3_smlad"
        packed = PackedConstant(
            name=packed_name,
            source=step.weight,
            layout="conv2d_3x3_arm_smlad_i16x2_v1",
            value=_pack_signed_i8_pairs(weight),
            alignment=4,
        )
        return KernelCapability(
            kernel_id=_CORTEX_M4_CONV_3X3_ID,
            priority=290,
            optimized=True,
            supported=True,
            reason=(
                "one-pixel static im2col patch feeds ARMv7E-M SMLAD; scratch is "
                "reused for every output pixel"
            ),
            packed_constants=(packed,),
            constant_overrides={step.weight: packed.name},
            scratch_size=4 * ((9 * input_channels + 1) // 2),
            scratch_alignment=4,
        )

    return (cortex_m4_1x1(), cortex_m4_3x3(), generic_1x1(), portable)


@kernel_capabilities.register
def _depthwise_capabilities(
    step: DepthwiseConv2DStep,
    plan: ExecutionPlan,
    options: CBackendOptions,
) -> tuple[KernelCapability, ...]:
    portable = KernelCapability(
        kernel_id=_PORTABLE_DEPTHWISE_ID,
        priority=0,
        optimized=False,
        supported=True,
        reason="generic NHWC/HWO DepthwiseConv2D kernel supports every verified step",
    )
    input_type = plan.tensors[step.input].tensor_type
    output_type = plan.tensors[step.output].tensor_type
    weight = plan.constants[step.weight]
    output_channels = output_type.shape[3]
    def generic_depthwise() -> KernelCapability:
        failure: str | None = None
        if not options.enable_weight_packing:
            failure = "weight packing is disabled"
        elif weight.dtype != np.int8 or weight.shape != (3, 3, output_channels):
            failure = "Depthwise optimized kernel requires a 3x3 HWO int8 weight"
        elif step.depth_multiplier != 1:
            failure = "Depthwise optimized kernel requires depth_multiplier one"
        elif output_channels < 2 or output_channels % 2:
            failure = "Depthwise optimized kernel requires an even output-channel pair shape"
        elif step.dilation != (1, 1):
            failure = "Depthwise 3x3 optimized kernel requires dilation (1, 1)"
        elif step.stride[0] not in (1, 2) or step.stride[1] not in (1, 2):
            failure = "Depthwise 3x3 optimized kernel supports only stride components one or two"
        if failure is not None:
            return _unsupported_capability(_DEPTHWISE_3X3_ID, failure)
        packed_name = f"{step.weight}.depthwise_3x3_c2"
        packed_value = np.ascontiguousarray(weight.reshape(9, output_channels // 2, 2))
        packed = PackedConstant(
            name=packed_name,
            source=step.weight,
            layout="depthwise_hwo_c2_interleaved_v1",
            value=packed_value,
        )
        return KernelCapability(
            kernel_id=_DEPTHWISE_3X3_ID,
            priority=100,
            optimized=True,
            supported=True,
            reason=(
                "3x3 HWO weights packed by channel pairs; one spatial input is "
                "reused across two channel accumulators"
            ),
            packed_constants=(packed,),
            constant_overrides={step.weight: packed.name},
        )

    def cortex_m4_depthwise() -> KernelCapability:
        failure: str | None = None
        if "dsp" not in options.target.features or "armv7e-m" not in options.target.features:
            failure = "target does not provide ARMv7E-M DSP instructions"
        elif not options.enable_weight_packing:
            failure = "weight packing is disabled"
        elif weight.dtype != np.int8 or weight.shape != (3, 3, output_channels):
            failure = "Cortex-M4 depthwise kernel requires 3x3 HWO int8 weights"
        elif step.depth_multiplier != 1 or step.dilation != (1, 1):
            failure = "Cortex-M4 depthwise v1 requires multiplier and dilation one"
        elif step.stride[0] not in (1, 2) or step.stride[1] not in (1, 2):
            failure = "Cortex-M4 depthwise v1 supports stride components one or two"
        if failure is not None:
            return _unsupported_capability(_CORTEX_M4_DEPTHWISE_ID, failure, priority=300)
        packed_name = f"{step.weight}.cortex_m4_smlad"
        packed = PackedConstant(
            name=packed_name,
            source=step.weight,
            layout="depthwise_3x3_arm_smlad_i16x2_v1",
            value=_pack_signed_i8_pairs(weight.transpose(2, 0, 1)),
            alignment=4,
        )
        return KernelCapability(
            kernel_id=_CORTEX_M4_DEPTHWISE_ID,
            priority=300,
            optimized=True,
            supported=True,
            reason="ARMv7E-M SMLAD evaluates two spatial depthwise taps per instruction",
            packed_constants=(packed,),
            constant_overrides={step.weight: packed.name},
        )

    return (cortex_m4_depthwise(), generic_depthwise(), portable)


def _step_constants(
    step: Conv2DStep | DepthwiseConv2DStep,
    context: StepEmitContext,
) -> tuple[str, str, tuple[ConstantEmission, ...]]:
    multiplier_symbol = f"{context.symbol}_op{context.step_index}_multiplier"
    shift_symbol = f"{context.symbol}_op{context.step_index}_shift"
    constants = (
        _int32_constant(multiplier_symbol, step.multipliers),
        _int32_constant(shift_symbol, step.shifts),
    )
    return multiplier_symbol, shift_symbol, constants


@emit_step.register
def _emit_conv2d(step: Conv2DStep, context: StepEmitContext) -> StepEmission:
    input_type = context.plan.tensors[step.input].tensor_type
    output_type = context.plan.tensors[step.output].tensor_type
    weight_type = context.plan.tensors[step.weight].tensor_type
    input_qparams = input_type.qparams
    output_qparams = output_type.qparams
    assert isinstance(input_qparams, PerTensorQParams)
    assert isinstance(output_qparams, PerTensorQParams)
    _, input_height, input_width, input_channels = input_type.shape
    output_channels, kernel_height, kernel_width, _ = weight_type.shape
    _, output_height, output_width, _ = output_type.shape
    stride_height, stride_width = step.stride
    dilation_height, dilation_width = step.dilation
    pad_top, _, pad_left, _ = step.padding
    multiplier_symbol, shift_symbol, constants = _step_constants(step, context)
    implementation = (
        _PORTABLE_CONV_ID if context.selection is None else context.selection.kernel_id
    )
    if implementation == _CORTEX_M4_CONV_1X1_ID:
        kernel_fn = f"{context.symbol}_conv2d_1x1_cortex_m4_smlad_s8"
        call = f"""    {kernel_fn}(
        {context.pointer(step.input, mutable=False)},
        {context.pointer(step.weight, mutable=False)},
        {context.pointer(step.bias, mutable=False)},
        {multiplier_symbol}, {shift_symbol},
        {context.pointer(step.output, mutable=True)},
        {input_width}u, {input_channels}u,
        {output_height}u, {output_width}u, {output_channels}u,
        {stride_height}u, {stride_width}u,
        {input_qparams.zero_point}, {output_qparams.zero_point},
        {step.activation_min}, {step.activation_max});"""
        selected_kernel = _cortex_m4_conv1x1_kernel(context)
    elif implementation == _CORTEX_M4_CONV_3X3_ID:
        kernel_fn = f"{context.symbol}_conv2d_3x3_cortex_m4_smlad_s8"
        call = f"""    {kernel_fn}(
        {context.pointer(step.input, mutable=False)},
        {context.pointer(step.weight, mutable=False)},
        {context.pointer(step.bias, mutable=False)},
        {multiplier_symbol}, {shift_symbol},
        {context.pointer(step.output, mutable=True)},
        (int32_t *){context.scratch_pointer},
        {input_height}u, {input_width}u, {input_channels}u,
        {output_height}u, {output_width}u, {output_channels}u,
        {pad_top}u, {pad_left}u,
        {input_qparams.zero_point}, {output_qparams.zero_point},
        {step.activation_min}, {step.activation_max});"""
        selected_kernel = _cortex_m4_conv3x3_kernel(context)
    elif implementation == _CONV_1X1_ID:
        kernel_fn = f"{context.symbol}_conv2d_1x1_o2_s8"
        call = f"""    {kernel_fn}(
        {context.pointer(step.input, mutable=False)},
        {context.pointer(step.weight, mutable=False)},
        {context.pointer(step.bias, mutable=False)},
        {multiplier_symbol}, {shift_symbol},
        {context.pointer(step.output, mutable=True)},
        {input_height}u, {input_width}u, {input_channels}u,
        {output_height}u, {output_width}u, {output_channels}u,
        {stride_height}u, {stride_width}u,
        {input_qparams.zero_point}, {output_qparams.zero_point},
        {step.activation_min}, {step.activation_max});"""
        selected_kernel = _conv1x1_o2_kernel(context)
    elif implementation == _PORTABLE_CONV_ID:
        kernel_fn = f"{context.symbol}_conv2d_s8"
        call = f"""    {kernel_fn}(
        {context.pointer(step.input, mutable=False)},
        {context.pointer(step.weight, mutable=False)},
        {context.pointer(step.bias, mutable=False)},
        {multiplier_symbol}, {shift_symbol},
        {context.pointer(step.output, mutable=True)},
        {input_height}u, {input_width}u, {input_channels}u,
        {output_height}u, {output_width}u, {output_channels}u,
        {step.groups}u,
        {kernel_height}u, {kernel_width}u,
        {stride_height}u, {stride_width}u,
        {dilation_height}u, {dilation_width}u,
        {pad_top}u, {pad_left}u,
        {input_qparams.zero_point}, {output_qparams.zero_point},
        {step.activation_min}, {step.activation_max});"""
        selected_kernel = _conv2d_kernel(context)
    else:
        raise CompileError(f"unsupported Conv2D C implementation {implementation}")
    return StepEmission(
        constants=constants,
        kernels=(q31_kernel(context), selected_kernel),
        call=call,
        manifest={
            "name": step.name,
            "kind": step.kernel_kind,
            "input": step.input,
            "output": step.output,
            "stride": list(step.stride),
            "dilation": list(step.dilation),
            "padding": list(step.padding),
            "groups": step.groups,
            "accumulator_bound_max": max(step.accumulator_bounds),
        },
    )


@emit_step.register
def _emit_depthwise_conv2d(
    step: DepthwiseConv2DStep,
    context: StepEmitContext,
) -> StepEmission:
    input_type = context.plan.tensors[step.input].tensor_type
    output_type = context.plan.tensors[step.output].tensor_type
    weight_type = context.plan.tensors[step.weight].tensor_type
    input_qparams = input_type.qparams
    output_qparams = output_type.qparams
    assert isinstance(input_qparams, PerTensorQParams)
    assert isinstance(output_qparams, PerTensorQParams)
    _, input_height, input_width, input_channels = input_type.shape
    kernel_height, kernel_width, output_channels = weight_type.shape
    _, output_height, output_width, _ = output_type.shape
    stride_height, stride_width = step.stride
    dilation_height, dilation_width = step.dilation
    pad_top, _, pad_left, _ = step.padding
    multiplier_symbol, shift_symbol, constants = _step_constants(step, context)
    implementation = (
        _PORTABLE_DEPTHWISE_ID
        if context.selection is None
        else context.selection.kernel_id
    )
    if implementation == _CORTEX_M4_DEPTHWISE_ID:
        kernel_fn = f"{context.symbol}_depthwise_3x3_cortex_m4_smlad_s8"
        call = f"""    {kernel_fn}(
        {context.pointer(step.input, mutable=False)},
        {context.pointer(step.weight, mutable=False)},
        {context.pointer(step.bias, mutable=False)},
        {multiplier_symbol}, {shift_symbol},
        {context.pointer(step.output, mutable=True)},
        {input_height}u, {input_width}u, {output_channels}u,
        {output_height}u, {output_width}u,
        {stride_height}u, {stride_width}u,
        {pad_top}u, {pad_left}u,
        {input_qparams.zero_point}, {output_qparams.zero_point},
        {step.activation_min}, {step.activation_max});"""
        selected_kernel = _cortex_m4_depthwise_kernel(context)
    elif implementation == _DEPTHWISE_3X3_ID:
        kernel_fn = f"{context.symbol}_depthwise_3x3_c2_s8"
        call = f"""    {kernel_fn}(
        {context.pointer(step.input, mutable=False)},
        {context.pointer(step.weight, mutable=False)},
        {context.pointer(step.bias, mutable=False)},
        {multiplier_symbol}, {shift_symbol},
        {context.pointer(step.output, mutable=True)},
        {input_height}u, {input_width}u, {input_channels}u,
        {output_height}u, {output_width}u, {output_channels}u,
        {stride_height}u, {stride_width}u,
        {pad_top}u, {pad_left}u,
        {input_qparams.zero_point}, {output_qparams.zero_point},
        {step.activation_min}, {step.activation_max});"""
        selected_kernel = _depthwise_3x3_c2_kernel(context)
    elif implementation == _PORTABLE_DEPTHWISE_ID:
        kernel_fn = f"{context.symbol}_depthwise_conv2d_s8"
        call = f"""    {kernel_fn}(
        {context.pointer(step.input, mutable=False)},
        {context.pointer(step.weight, mutable=False)},
        {context.pointer(step.bias, mutable=False)},
        {multiplier_symbol}, {shift_symbol},
        {context.pointer(step.output, mutable=True)},
        {input_height}u, {input_width}u, {input_channels}u,
        {output_height}u, {output_width}u, {output_channels}u,
        {kernel_height}u, {kernel_width}u, {step.depth_multiplier}u,
        {stride_height}u, {stride_width}u,
        {dilation_height}u, {dilation_width}u,
        {pad_top}u, {pad_left}u,
        {input_qparams.zero_point}, {output_qparams.zero_point},
        {step.activation_min}, {step.activation_max});"""
        selected_kernel = _depthwise_kernel(context)
    else:
        raise CompileError(
            f"unsupported DepthwiseConv2D C implementation {implementation}"
        )
    return StepEmission(
        constants=constants,
        kernels=(q31_kernel(context), selected_kernel),
        call=call,
        manifest={
            "name": step.name,
            "kind": step.kernel_kind,
            "input": step.input,
            "output": step.output,
            "depth_multiplier": step.depth_multiplier,
            "stride": list(step.stride),
            "dilation": list(step.dilation),
            "padding": list(step.padding),
            "accumulator_bound_max": max(step.accumulator_bounds),
        },
    )


__all__: list[str] = []
