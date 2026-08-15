from __future__ import annotations

from math import prod

import numpy as np

from bakenn.backend.portable_c.contracts import KernelEmission, StepEmitContext
from bakenn.backend.portable_c.selection import (
    CBackendOptions,
    KernelCapability,
    PackedConstant,
)
from bakenn.ir import PerTensorQParams
from bakenn.ir.types import TARGET_SIZE_MAX
from bakenn.plan import ExecutionPlan, LinearStep
from bakenn.plan.steps import (
    AveragePool2DStep,
    Conv2DStep,
    DepthwiseConv2DStep,
    MaxPool2DStep,
)

from .bundle import ESP_NN_VERSION


ESP_NN_CONV_IDS = {
    target: f"esp_nn.{target}.conv2d_s8.v{ESP_NN_VERSION}"
    for target in ("esp32", "esp32s3")
}
ESP_NN_DEPTHWISE_IDS = {
    target: f"esp_nn.{target}.depthwise_conv2d_s8.v{ESP_NN_VERSION}"
    for target in ("esp32", "esp32s3")
}
ESP_NN_LINEAR_IDS = {
    target: f"esp_nn.{target}.linear_per_channel_s8.v{ESP_NN_VERSION}"
    for target in ("esp32", "esp32s3")
}
ESP_NN_AVERAGE_POOL_IDS = {
    target: f"esp_nn.{target}.average_pool2d_s8.v{ESP_NN_VERSION}"
    for target in ("esp32", "esp32s3")
}
ESP_NN_MAX_POOL_IDS = {
    target: f"esp_nn.{target}.max_pool2d_s8.v{ESP_NN_VERSION}"
    for target in ("esp32", "esp32s3")
}

_ESP_TARGETS = frozenset({"esp32", "esp32s3"})
_I32_MAX = (1 << 31) - 1
_U16_MAX = (1 << 16) - 1
_MIN_S3_LINEAR_MACS = 256


def is_esp_nn_kernel(kernel_id: str) -> bool:
    return kernel_id.startswith("esp_nn.")


def _unsupported(kernel_id: str, reason: str) -> KernelCapability:
    return KernelCapability(
        kernel_id=kernel_id,
        priority=500,
        optimized=True,
        supported=False,
        reason=reason,
    )


def _target_id(options: CBackendOptions) -> str:
    return options.target.target_id


def _base_failure(options: CBackendOptions) -> str | None:
    if not options.enable_esp_nn:
        return "ESP-NN source bundling is disabled"
    if _target_id(options) not in _ESP_TARGETS:
        return "ESP-NN v1.2.6 requires target esp32 or esp32s3"
    return None


def _shape_fits(*shapes: tuple[int, ...], parameters: tuple[int, ...] = ()) -> bool:
    values = (*parameters, *(value for shape in shapes for value in shape))
    return (
        all(0 <= value <= _U16_MAX for value in values)
        and all(prod(shape) <= _I32_MAX for shape in shapes)
    )


def _average_pool_rounding_is_exact(
    step: AveragePool2DStep,
    plan: ExecutionPlan,
) -> bool:
    input_type = plan.tensors[step.input].tensor_type
    output_type = plan.tensors[step.output].tensor_type
    qparams = input_type.qparams
    assert isinstance(qparams, PerTensorQParams)
    if qparams.zero_point == 0:
        return True
    _, input_height, input_width, _ = input_type.shape
    _, output_height, output_width, _ = output_type.shape
    kernel_height, kernel_width = step.kernel
    stride_height, stride_width = step.stride
    pad_top, _, pad_left, _ = step.padding
    for output_y in range(output_height):
        start_y = output_y * stride_height - pad_top
        valid_height = max(
            0,
            min(start_y + kernel_height, input_height) - max(start_y, 0),
        )
        for output_x in range(output_width):
            start_x = output_x * stride_width - pad_left
            valid_width = max(
                0,
                min(start_x + kernel_width, input_width) - max(start_x, 0),
            )
            if (valid_height * valid_width) % 2 == 0:
                return False
    return True


def _esp32s3_conv_scratch(
    step: Conv2DStep,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
) -> int:
    _, input_height, input_width, input_channels = input_shape
    _, output_height, output_width, output_channels = output_shape
    _, filter_height, filter_width, _ = weight_shape
    pad_top, _, pad_left, _ = step.padding
    stride_height, stride_width = step.stride
    new_channels = (input_channels + 7) & ~7
    input_scratch = input_width * input_height * input_channels
    filter_scratch = (
        filter_width * filter_height * input_channels * output_channels
    )
    alignment_margin = 64

    if (
        filter_width == filter_height == 1
        and pad_left == pad_top == 0
        and stride_width == stride_height == 1
    ):
        transpose = 16 * new_channels if input_width * input_height >= 8 else 0
        input_scratch = (
            input_width * input_height * new_channels
            if input_channels % 8
            else 0
        )
        filter_scratch = new_channels * output_channels
        return input_scratch + filter_scratch + transpose + alignment_margin

    filter_row = filter_width * input_channels
    window = filter_width * filter_height * input_channels
    if filter_row < 16 and window >= 16:
        aligned_window = (window + 15) & ~15
        return (
            output_channels * 4
            + 16
            + output_channels * aligned_window
            + 16
            + aligned_window
            + alignment_margin
        )

    new_channels = (input_channels + 15) & ~15
    pad_right = max(
        0,
        (output_width - 1) * stride_width
        + filter_width
        - pad_left
        - input_width,
    )
    pad_bottom = max(
        0,
        (output_height - 1) * stride_height
        + filter_height
        - pad_top
        - input_height,
    )
    if pad_left == pad_top == pad_right == pad_bottom == 0:
        input_scratch = 0
    else:
        input_scratch = (
            (input_width + pad_left + pad_right)
            * (input_height + pad_top + pad_bottom)
            * input_channels
        )
    filter_scratch = (
        filter_width * filter_height * new_channels * output_channels
    )
    aligned_filter_row = ((filter_row + 15) // 16) * 16
    alignment_scratch = aligned_filter_row * filter_height * output_channels
    return (
        input_scratch
        + filter_scratch
        + alignment_scratch
        + alignment_margin
        + output_channels * 4
    )


def _esp32s3_depthwise_scratch(
    step: DepthwiseConv2DStep,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
) -> int:
    _, input_height, input_width, channels = input_shape
    _, output_height, output_width, _ = output_shape
    filter_height, filter_width, _ = weight_shape
    pad_top, _, pad_left, _ = step.padding
    stride_height, stride_width = step.stride
    multiplier = step.depth_multiplier
    filter_size = filter_width * filter_height * channels * multiplier

    if multiplier == 1 and channels % 8 == 0:
        if filter_width == filter_height == 3:
            if channels % 16 == 0:
                if pad_left or pad_top:
                    pad_width = pad_left * 2
                    pad_height = pad_top * 2
                else:
                    pad_width = (
                        output_width * stride_width + filter_width - 1
                    ) - input_width
                    pad_height = (
                        output_height * stride_height + filter_height - 1
                    ) - input_height
                if pad_width or pad_height:
                    full_input = (
                        (input_width + pad_width)
                        * (input_height + pad_height)
                        * channels
                    )
                    if full_input <= 40 * 1024:
                        return filter_size + full_input + 16
                    strip = (
                        (input_width + pad_width) * filter_height * channels
                    )
                    return filter_size + strip + 16
                return filter_size + 16
            if channels >= 12:
                padded_channels = (channels + 15) & ~15
                padded_filter = 9 * padded_channels
                total_pad_width = pad_left * 2 + max(
                    0,
                    output_width * stride_width + 2 - input_width,
                )
                total_pad_height = pad_top * 2 + max(
                    0,
                    output_height * stride_height + 2 - input_height,
                )
                padded_input = (
                    (input_width + total_pad_width)
                    * (input_height + total_pad_height)
                    * padded_channels
                )
                output_buffer = output_width * output_height * padded_channels
                return padded_filter + padded_input + output_buffer + 64
            return 2 * (
                filter_size + input_width * input_height * channels
            ) + 32

        full_s16 = 2 * (
            filter_size + input_width * input_height * channels
        )
        if full_s16 <= 48 * 1024:
            return full_s16 + 32
        tile_s16 = 2 * input_width * filter_height * channels
        return 2 * filter_size + tile_s16 + 32

    if multiplier == 1 and channels > 3:
        padded_channels = (channels + 7) & ~7
        filter_bytes = filter_width * filter_height * padded_channels * 2
        input_start = (filter_bytes + 15) & ~15
        input_bytes = input_width * input_height * padded_channels * 2
        output_start = (input_start + input_bytes + 15) & ~15
        output_bytes = output_width * output_height * padded_channels
        bias_start = (output_start + output_bytes + 15) & ~15
        channel_i32 = padded_channels * 4
        return bias_start + channel_i32 * 3 + 16

    if multiplier % 4 == 0:
        input_size = input_width * input_height * channels
        return 2 * (filter_size + input_size) + 32
    return 32


def conv_capability(
    step: Conv2DStep,
    plan: ExecutionPlan,
    options: CBackendOptions,
) -> KernelCapability:
    target = _target_id(options)
    kernel_id = ESP_NN_CONV_IDS.get(target, ESP_NN_CONV_IDS["esp32s3"])
    failure = _base_failure(options)
    input_type = plan.tensors[step.input].tensor_type
    output_type = plan.tensors[step.output].tensor_type
    weight = plan.constants[step.weight]
    if failure is None and step.groups != 1:
        failure = "ESP-NN optimized Conv2D requires groups one"
    elif failure is None and step.dilation != (1, 1):
        failure = "ESP-NN Conv2D ignores dilation other than one"
    elif failure is None and not _shape_fits(
        input_type.shape,
        output_type.shape,
        weight.shape,
        parameters=(*step.stride, *step.padding),
    ):
        failure = "ESP-NN Conv2D dimensions must fit uint16 fields and int32 products"
    elif failure is None and (
        weight.dtype != np.int8
        or weight.ndim != 4
        or weight.shape[0] != output_type.shape[3]
        or weight.shape[3] != input_type.shape[3]
    ):
        failure = "ESP-NN Conv2D requires matching OHWI int8 weights"
    if failure is not None:
        return _unsupported(kernel_id, failure)
    scratch = (
        _esp32s3_conv_scratch(step, input_type.shape, output_type.shape, weight.shape)
        if target == "esp32s3"
        else 0
    )
    if scratch > min(TARGET_SIZE_MAX, _I32_MAX):
        return _unsupported(kernel_id, "ESP-NN Conv2D scratch exceeds int32 storage")
    return KernelCapability(
        kernel_id=kernel_id,
        priority=500,
        optimized=True,
        supported=True,
        reason=(
            "pinned ESP-NN v1.2.6 selects the ESP32-S3 SIMD/assembly Conv2D path"
            if target == "esp32s3"
            else "pinned ESP-NN v1.2.6 selects its ESP32 generic optimized Conv2D path"
        ),
        scratch_size=scratch,
        scratch_alignment=16 if scratch else 1,
    )


def depthwise_capability(
    step: DepthwiseConv2DStep,
    plan: ExecutionPlan,
    options: CBackendOptions,
) -> KernelCapability:
    target = _target_id(options)
    kernel_id = ESP_NN_DEPTHWISE_IDS.get(
        target, ESP_NN_DEPTHWISE_IDS["esp32s3"]
    )
    failure = _base_failure(options)
    input_type = plan.tensors[step.input].tensor_type
    output_type = plan.tensors[step.output].tensor_type
    weight = plan.constants[step.weight]
    if failure is None and step.dilation != (1, 1):
        failure = "ESP-NN DepthwiseConv2D ignores dilation other than one"
    elif failure is None and target == "esp32s3" and (
        step.padding[0] != step.padding[1]
        or step.padding[2] != step.padding[3]
    ):
        failure = "ESP32-S3 DepthwiseConv2D scratch path requires symmetric padding"
    elif failure is None and not _shape_fits(
        input_type.shape,
        output_type.shape,
        weight.shape,
        parameters=(*step.stride, *step.padding, step.depth_multiplier),
    ):
        failure = (
            "ESP-NN DepthwiseConv2D dimensions must fit uint16 fields and int32 products"
        )
    elif failure is None and (
        weight.dtype != np.int8
        or weight.ndim != 3
        or weight.shape[2] != output_type.shape[3]
        or output_type.shape[3]
        != input_type.shape[3] * step.depth_multiplier
    ):
        failure = "ESP-NN DepthwiseConv2D requires matching HWO int8 weights"
    if failure is not None:
        return _unsupported(kernel_id, failure)
    scratch = (
        _esp32s3_depthwise_scratch(
            step, input_type.shape, output_type.shape, weight.shape
        )
        if target == "esp32s3"
        else 0
    )
    if scratch > min(TARGET_SIZE_MAX, _I32_MAX):
        return _unsupported(
            kernel_id, "ESP-NN DepthwiseConv2D scratch exceeds int32 storage"
        )
    return KernelCapability(
        kernel_id=kernel_id,
        priority=500,
        optimized=True,
        supported=True,
        reason=(
            "pinned ESP-NN v1.2.6 selects the ESP32-S3 SIMD/assembly depthwise path"
            if target == "esp32s3"
            else "pinned ESP-NN v1.2.6 selects its ESP32 generic optimized depthwise path"
        ),
        scratch_size=scratch,
        scratch_alignment=16 if scratch else 1,
    )


def linear_capability(
    step: LinearStep,
    plan: ExecutionPlan,
    options: CBackendOptions,
) -> KernelCapability:
    target = _target_id(options)
    kernel_id = ESP_NN_LINEAR_IDS.get(target, ESP_NN_LINEAR_IDS["esp32s3"])
    failure = _base_failure(options)
    input_type = plan.tensors[step.input].tensor_type
    output_type = plan.tensors[step.output].tensor_type
    weight = plan.constants[step.weight]
    input_count = input_type.shape[1]
    output_count = output_type.shape[1]
    if failure is None and target != "esp32s3":
        failure = "ESP-NN maps ESP32 FullyConnected to ANSI C; BakeNN keeps its own kernel"
    elif failure is None and not options.enable_weight_packing:
        failure = "safe ESP32-S3 FullyConnected requires host-side guard packing"
    elif failure is None and (
        input_count > _U16_MAX or output_count > _U16_MAX
    ):
        failure = "ESP-NN FullyConnected dimensions must fit uint16 fields"
    elif failure is None and input_count * output_count < _MIN_S3_LINEAR_MACS:
        failure = (
            f"ESP32-S3 FullyConnected MAC count is below {_MIN_S3_LINEAR_MACS}"
        )
    elif failure is None and (
        weight.dtype != np.int8
        or weight.shape != (output_count, input_count)
    ):
        failure = "ESP-NN FullyConnected requires matching OI int8 weights"
    if failure is not None:
        return _unsupported(kernel_id, failure)

    padded_count = (input_count + 15) & ~15
    packed_value = np.zeros(output_count * padded_count + 32, dtype=np.int8)
    packed_rows = packed_value[: output_count * padded_count].reshape(
        output_count, padded_count
    )
    packed_rows[:, :input_count] = weight
    packed_name = f"{step.weight}.esp32s3_fc_guarded"
    packed = PackedConstant(
        name=packed_name,
        source=step.weight,
        layout="esp_nn_fc_oi_row16_guard32_v1",
        value=packed_value,
        alignment=16,
    )
    scratch = padded_count + 32
    return KernelCapability(
        kernel_id=kernel_id,
        priority=500,
        optimized=True,
        supported=True,
        reason=(
            "ESP32-S3 SIMD per-channel FullyConnected with host-packed aligned weights "
            "and a bounded guarded input staging buffer"
        ),
        packed_constants=(packed,),
        constant_overrides={step.weight: packed.name},
        scratch_size=scratch,
        scratch_alignment=16,
    )


def pool_capability(
    step: AveragePool2DStep | MaxPool2DStep,
    plan: ExecutionPlan,
    options: CBackendOptions,
) -> KernelCapability:
    target = _target_id(options)
    identifiers = (
        ESP_NN_AVERAGE_POOL_IDS
        if isinstance(step, AveragePool2DStep)
        else ESP_NN_MAX_POOL_IDS
    )
    kernel_id = identifiers.get(target, identifiers["esp32s3"])
    failure = _base_failure(options)
    input_type = plan.tensors[step.input].tensor_type
    output_type = plan.tensors[step.output].tensor_type
    if failure is None and target != "esp32s3":
        failure = "ESP-NN maps ESP32 pooling to ANSI C; BakeNN keeps its own kernel"
    elif failure is None and input_type.shape[3] % 4:
        failure = "ESP32-S3 pooling assembly requires channels divisible by four"
    elif (
        failure is None
        and isinstance(step, AveragePool2DStep)
        and not _average_pool_rounding_is_exact(step, plan)
    ):
        failure = (
            "ESP-NN AveragePool raw-code rounding can differ from BakeNN centered "
            "half-away rounding for this zero-point/window combination"
        )
    elif failure is None and not _shape_fits(
        input_type.shape,
        output_type.shape,
        parameters=(*step.kernel, *step.stride, *step.padding),
    ):
        failure = "ESP-NN pooling dimensions must fit uint16 fields and int32 products"
    if failure is not None:
        return _unsupported(kernel_id, failure)
    return KernelCapability(
        kernel_id=kernel_id,
        priority=500,
        optimized=True,
        supported=True,
        reason=(
            "pinned ESP-NN v1.2.6 selects ESP32-S3 SIMD average pooling"
            if isinstance(step, AveragePool2DStep)
            else "pinned ESP-NN v1.2.6 selects ESP32-S3 SIMD max pooling"
        ),
    )


def _scratch_expression(context: StepEmitContext) -> str:
    assert context.selection is not None
    return context.scratch_pointer if context.selection.scratch_size else "NULL"


def conv_emission(
    step: Conv2DStep,
    context: StepEmitContext,
    multiplier_symbol: str,
    shift_symbol: str,
) -> tuple[KernelEmission, str]:
    input_type = context.plan.tensors[step.input].tensor_type
    output_type = context.plan.tensors[step.output].tensor_type
    weight = context.plan.constants[step.weight]
    input_qparams = input_type.qparams
    output_qparams = output_type.qparams
    assert isinstance(input_qparams, PerTensorQParams)
    assert isinstance(output_qparams, PerTensorQParams)
    function = f"{context.symbol}_conv2d_esp_nn_s8"
    signature = f"""void {function}(
    const int8_t *input, const int8_t *weight, const int32_t *bias,
    const int32_t *multiplier, const int32_t *shift, int8_t *output, void *scratch,
    int32_t input_height, int32_t input_width, int32_t input_channels,
    int32_t output_height, int32_t output_width, int32_t output_channels,
    int32_t kernel_height, int32_t kernel_width,
    int32_t stride_height, int32_t stride_width,
    int32_t pad_top, int32_t pad_left,
    int32_t input_zero_point, int32_t output_zero_point,
    int32_t activation_min, int32_t activation_max)"""
    kernel = KernelEmission(
        key="esp_nn_conv2d_s8_v1_2_6",
        header_includes=("<stddef.h>", "<stdint.h>", '"esp_nn.h"'),
        declaration=signature + ";",
        definition=f"""{signature} {{
    const data_dims_t input_dims = {{ input_width, input_height, input_channels, 1 }};
    const data_dims_t filter_dims = {{ kernel_width, kernel_height, input_channels, output_channels }};
    const data_dims_t output_dims = {{ output_width, output_height, output_channels, 1 }};
    const conv_params_t parameters = {{
        .in_offset = -input_zero_point,
        .out_offset = output_zero_point,
        .stride = {{ stride_width, stride_height }},
        .padding = {{ pad_left, pad_top }},
        .dilation = {{ 1, 1 }},
        .activation = {{ activation_min, activation_max }},
    }};
    const quant_data_t quantization = {{
        .shift = (int32_t *)(uintptr_t)shift,
        .mult = (int32_t *)(uintptr_t)multiplier,
    }};
    esp_nn_set_conv_scratch_buf(scratch);
    esp_nn_conv_s8(&input_dims, input, &filter_dims, weight, bias,
                   &output_dims, output, &parameters, &quantization);
}}""",
    )
    call = f"""    {function}(
        {context.pointer(step.input, mutable=False)},
        {context.pointer(step.weight, mutable=False)},
        {context.pointer(step.bias, mutable=False)},
        {multiplier_symbol}, {shift_symbol},
        {context.pointer(step.output, mutable=True)}, {_scratch_expression(context)},
        {input_type.shape[1]}, {input_type.shape[2]}, {input_type.shape[3]},
        {output_type.shape[1]}, {output_type.shape[2]}, {output_type.shape[3]},
        {weight.shape[1]}, {weight.shape[2]},
        {step.stride[0]}, {step.stride[1]}, {step.padding[0]}, {step.padding[2]},
        {input_qparams.zero_point}, {output_qparams.zero_point},
        {step.activation_min}, {step.activation_max});"""
    return kernel, call


def depthwise_emission(
    step: DepthwiseConv2DStep,
    context: StepEmitContext,
    multiplier_symbol: str,
    shift_symbol: str,
) -> tuple[KernelEmission, str]:
    input_type = context.plan.tensors[step.input].tensor_type
    output_type = context.plan.tensors[step.output].tensor_type
    weight = context.plan.constants[step.weight]
    input_qparams = input_type.qparams
    output_qparams = output_type.qparams
    assert isinstance(input_qparams, PerTensorQParams)
    assert isinstance(output_qparams, PerTensorQParams)
    function = f"{context.symbol}_depthwise_conv2d_esp_nn_s8"
    signature = f"""void {function}(
    const int8_t *input, const int8_t *weight, const int32_t *bias,
    const int32_t *multiplier, const int32_t *shift, int8_t *output, void *scratch,
    int32_t input_height, int32_t input_width, int32_t input_channels,
    int32_t output_height, int32_t output_width, int32_t output_channels,
    int32_t kernel_height, int32_t kernel_width, int32_t depth_multiplier,
    int32_t stride_height, int32_t stride_width,
    int32_t pad_top, int32_t pad_left,
    int32_t input_zero_point, int32_t output_zero_point,
    int32_t activation_min, int32_t activation_max)"""
    kernel = KernelEmission(
        key="esp_nn_depthwise_conv2d_s8_v1_2_6",
        header_includes=("<stddef.h>", "<stdint.h>", '"esp_nn.h"'),
        declaration=signature + ";",
        definition=f"""{signature} {{
    const data_dims_t input_dims = {{ input_width, input_height, input_channels, 1 }};
    const data_dims_t filter_dims = {{ kernel_width, kernel_height, output_channels, 1 }};
    const data_dims_t output_dims = {{ output_width, output_height, output_channels, 1 }};
    const dw_conv_params_t parameters = {{
        .in_offset = -input_zero_point,
        .out_offset = output_zero_point,
        .ch_mult = depth_multiplier,
        .stride = {{ stride_width, stride_height }},
        .padding = {{ pad_left, pad_top }},
        .dilation = {{ 1, 1 }},
        .activation = {{ activation_min, activation_max }},
    }};
    const quant_data_t quantization = {{
        .shift = (int32_t *)(uintptr_t)shift,
        .mult = (int32_t *)(uintptr_t)multiplier,
    }};
    esp_nn_set_depthwise_conv_scratch_buf(scratch);
    esp_nn_depthwise_conv_s8(&input_dims, input, &filter_dims, weight, bias,
                             &output_dims, output, &parameters, &quantization);
}}""",
    )
    call = f"""    {function}(
        {context.pointer(step.input, mutable=False)},
        {context.pointer(step.weight, mutable=False)},
        {context.pointer(step.bias, mutable=False)},
        {multiplier_symbol}, {shift_symbol},
        {context.pointer(step.output, mutable=True)}, {_scratch_expression(context)},
        {input_type.shape[1]}, {input_type.shape[2]}, {input_type.shape[3]},
        {output_type.shape[1]}, {output_type.shape[2]}, {output_type.shape[3]},
        {weight.shape[0]}, {weight.shape[1]}, {step.depth_multiplier},
        {step.stride[0]}, {step.stride[1]}, {step.padding[0]}, {step.padding[2]},
        {input_qparams.zero_point}, {output_qparams.zero_point},
        {step.activation_min}, {step.activation_max});"""
    return kernel, call


def linear_emission(
    step: LinearStep,
    context: StepEmitContext,
    multiplier_symbol: str,
    shift_symbol: str,
) -> tuple[KernelEmission, str]:
    input_type = context.plan.tensors[step.input].tensor_type
    output_type = context.plan.tensors[step.output].tensor_type
    input_qparams = input_type.qparams
    output_qparams = output_type.qparams
    assert isinstance(input_qparams, PerTensorQParams)
    assert isinstance(output_qparams, PerTensorQParams)
    padded_count = (input_type.shape[1] + 15) & ~15
    function = f"{context.symbol}_linear_esp_nn_s8"
    signature = f"""void {function}(
    const int8_t *input, const int8_t *weight, const int32_t *bias,
    const int32_t *multiplier, const int32_t *shift,
    int8_t *output, int8_t *staged_input,
    uint16_t input_count, uint16_t padded_count, uint16_t output_count,
    int32_t input_zero_point, int32_t output_zero_point,
    int32_t activation_min, int32_t activation_max)"""
    kernel = KernelEmission(
        key="esp_nn_linear_per_channel_s8_v1_2_6",
        header_includes=("<stdint.h>", '"esp_nn.h"'),
        declaration=signature + ";",
        definition=f"""{signature} {{
    for (uint16_t index = 0; index < input_count; ++index) {{
        staged_input[index] = input[index];
    }}
    for (uint16_t index = input_count; index < (uint16_t)(padded_count + 32u); ++index) {{
        staged_input[index] = (int8_t)input_zero_point;
    }}
    esp_nn_fully_connected_per_ch_s8(
        staged_input, -input_zero_point, padded_count, weight, 0, bias,
        output, output_count, output_zero_point,
        (int32_t *)(uintptr_t)shift, (int32_t *)(uintptr_t)multiplier,
        activation_min, activation_max);
}}""",
    )
    call = f"""    {function}(
        {context.pointer(step.input, mutable=False)},
        {context.pointer(step.weight, mutable=False)},
        {context.pointer(step.bias, mutable=False)},
        {multiplier_symbol}, {shift_symbol},
        {context.pointer(step.output, mutable=True)},
        (int8_t *){context.scratch_pointer},
        {input_type.shape[1]}u, {padded_count}u, {output_type.shape[1]}u,
        {input_qparams.zero_point}, {output_qparams.zero_point},
        {step.activation_min}, {step.activation_max});"""
    return kernel, call


def pool_emission(
    step: AveragePool2DStep | MaxPool2DStep,
    context: StepEmitContext,
) -> tuple[KernelEmission, str]:
    input_type = context.plan.tensors[step.input].tensor_type
    output_type = context.plan.tensors[step.output].tensor_type
    average = isinstance(step, AveragePool2DStep)
    operation = "average_pool2d" if average else "max_pool2d"
    api = "esp_nn_avg_pool_s8" if average else "esp_nn_max_pool_s8"
    function = f"{context.symbol}_{operation}_esp_nn_s8"
    signature = f"""void {function}(
    const int8_t *input, int8_t *output,
    uint16_t input_width, uint16_t input_height,
    uint16_t output_width, uint16_t output_height,
    uint16_t stride_width, uint16_t stride_height,
    uint16_t kernel_width, uint16_t kernel_height,
    uint16_t pad_left, uint16_t pad_top,
    int32_t activation_min, int32_t activation_max, uint16_t channels)"""
    kernel = KernelEmission(
        key=f"esp_nn_{operation}_s8_v1_2_6",
        header_includes=("<stdint.h>", '"esp_nn.h"'),
        declaration=signature + ";",
        definition=f"""{signature} {{
    {api}(input, input_width, input_height, output,
             output_width, output_height, stride_width, stride_height,
             kernel_width, kernel_height, pad_left, pad_top,
             activation_min, activation_max, channels);
}}""",
    )
    call = f"""    {function}(
        {context.pointer(step.input, mutable=False)},
        {context.pointer(step.output, mutable=True)},
        {input_type.shape[2]}u, {input_type.shape[1]}u,
        {output_type.shape[2]}u, {output_type.shape[1]}u,
        {step.stride[1]}u, {step.stride[0]}u,
        {step.kernel[1]}u, {step.kernel[0]}u,
        {step.padding[2]}u, {step.padding[0]}u,
        {step.activation_min}, {step.activation_max}, {input_type.shape[3]}u);"""
    return kernel, call


__all__ = [
    "ESP_NN_AVERAGE_POOL_IDS",
    "ESP_NN_CONV_IDS",
    "ESP_NN_DEPTHWISE_IDS",
    "ESP_NN_LINEAR_IDS",
    "ESP_NN_MAX_POOL_IDS",
    "conv_capability",
    "conv_emission",
    "depthwise_capability",
    "depthwise_emission",
    "is_esp_nn_kernel",
    "linear_capability",
    "linear_emission",
    "pool_capability",
    "pool_emission",
]
