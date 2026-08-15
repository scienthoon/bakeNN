from __future__ import annotations

from math import prod

from bakenn.backend.esp_nn.integration import (
    ESP_NN_AVERAGE_POOL_IDS,
    ESP_NN_MAX_POOL_IDS,
    pool_capability as esp_nn_pool_capability,
    pool_emission as esp_nn_pool_emission,
)
from bakenn.ir.types import PerTensorQParams
from bakenn.errors import CompileError
from bakenn.plan import ExecutionPlan
from bakenn.plan.steps.pool import AveragePool2DStep, MaxPool2DStep

from ..contracts import KernelEmission, StepEmitContext, StepEmission, emit_step
from ..selection import CBackendOptions, KernelCapability, kernel_capabilities


_PORTABLE_AVERAGE_ID = "portable.average_pool2d_s8.v1"
_PORTABLE_MAX_ID = "portable.max_pool2d_s8.v1"
_CORTEX_M4_GLOBAL_AVERAGE_ID = "cortex_m4.global_average_pool2d_s8.v1"
_CORTEX_M4_MAX_2X2_ID = "cortex_m4.max_pool2d_2x2_s2.v1"
_CMSIS_NN_AVERAGE_ID = "cmsis_nn.average_pool2d_s8.v4.0.0"
_CMSIS_NN_MAX_ID = "cmsis_nn.max_pool2d_s8.v4.0.0"
_CMSIS_I32_MAX = (1 << 31) - 1


def _target_supported(options: CBackendOptions) -> bool:
    return "dsp" in options.target.features and "armv7e-m" in options.target.features


def _symmetric_padding(padding: tuple[int, int, int, int]) -> bool:
    top, bottom, left, right = padding
    return top == bottom and left == right


def _cmsis_pool_shape_fits(
    step: AveragePool2DStep | MaxPool2DStep,
    plan: ExecutionPlan,
) -> bool:
    input_type = plan.tensors[step.input].tensor_type
    output_type = plan.tensors[step.output].tensor_type
    values = (
        *input_type.shape,
        *output_type.shape,
        *step.kernel,
        *step.stride,
        *step.padding,
    )
    return (
        all(0 <= value <= _CMSIS_I32_MAX for value in values)
        and prod(input_type.shape) <= _CMSIS_I32_MAX
        and prod(output_type.shape) <= _CMSIS_I32_MAX
        and prod(step.kernel) <= _CMSIS_I32_MAX
    )


def _cmsis_average_rounding_is_exact(
    step: AveragePool2DStep,
    plan: ExecutionPlan,
) -> bool:
    input_type = plan.tensors[step.input].tensor_type
    output_type = plan.tensors[step.output].tensor_type
    qparams = input_type.qparams
    assert isinstance(qparams, PerTensorQParams)
    if qparams.zero_point == 0:
        return True
    _, input_h, input_w, _ = input_type.shape
    _, output_h, output_w, _ = output_type.shape
    kernel_h, kernel_w = step.kernel
    stride_h, stride_w = step.stride
    pad_top, _, pad_left, _ = step.padding
    for output_y in range(output_h):
        start_y = output_y * stride_h - pad_top
        valid_h = max(0, min(start_y + kernel_h, input_h) - max(start_y, 0))
        for output_x in range(output_w):
            start_x = output_x * stride_w - pad_left
            valid_w = max(0, min(start_x + kernel_w, input_w) - max(start_x, 0))
            if (valid_h * valid_w) % 2 == 0:
                return False
    return True


@kernel_capabilities.register
def _average_capabilities(
    step: AveragePool2DStep,
    plan: ExecutionPlan,
    options: CBackendOptions,
) -> tuple[KernelCapability, ...]:
    input_type = plan.tensors[step.input].tensor_type
    output_type = plan.tensors[step.output].tensor_type
    global_shape = (
        step.padding == (0, 0, 0, 0)
        and step.kernel == input_type.shape[1:3]
        and output_type.shape[1:3] == (1, 1)
    )
    cmsis_supported = (
        options.enable_cmsis_nn
        and _target_supported(options)
        and _symmetric_padding(step.padding)
        and _cmsis_pool_shape_fits(step, plan)
        and input_type.shape[3] * 4 <= _CMSIS_I32_MAX
        and _cmsis_average_rounding_is_exact(step, plan)
    )
    cmsis = KernelCapability(
        _CMSIS_NN_AVERAGE_ID,
        400,
        True,
        cmsis_supported,
        (
            "pinned CMSIS-NN v4 AveragePool supports this ARMv7E-M DSP shape"
            if cmsis_supported
            else (
                "CMSIS-NN AveragePool requires source bundling, an ARMv7E-M DSP "
                "target, signed-32-bit dimensions/scratch, symmetric padding, and "
                "a zero-point/window combination that preserves BakeNN v1 "
                "half-away rounding"
            )
        ),
        scratch_size=input_type.shape[3] * 4 if cmsis_supported else 0,
        scratch_alignment=4 if cmsis_supported else 1,
    )
    optimized = KernelCapability(
        _CORTEX_M4_GLOBAL_AVERAGE_ID,
        200,
        True,
        _target_supported(options) and global_shape,
        (
            "fixed global spatial extent removes coordinate and valid-count branches"
            if _target_supported(options) and global_shape
            else "requires Cortex-M4 DSP target and exact global zero-padding shape"
        ),
    )
    portable = KernelCapability(
        _PORTABLE_AVERAGE_ID,
        0,
        False,
        True,
        "portable AveragePool2D supports every verified step",
    )
    return esp_nn_pool_capability(step, plan, options), cmsis, optimized, portable


@kernel_capabilities.register
def _max_capabilities(
    step: MaxPool2DStep,
    plan: ExecutionPlan,
    options: CBackendOptions,
) -> tuple[KernelCapability, ...]:
    cmsis_supported = (
        options.enable_cmsis_nn
        and _target_supported(options)
        and _symmetric_padding(step.padding)
        and _cmsis_pool_shape_fits(step, plan)
    )
    cmsis = KernelCapability(
        _CMSIS_NN_MAX_ID,
        400,
        True,
        cmsis_supported,
        (
            "pinned CMSIS-NN v4 MaxPool supports this ARMv7E-M DSP shape"
            if cmsis_supported
            else (
                "CMSIS-NN MaxPool requires source bundling, an ARMv7E-M DSP "
                "target, signed-32-bit dimensions, and symmetric padding"
            )
        ),
    )
    exact = step.kernel == (2, 2) and step.stride == (2, 2) and step.padding == (0, 0, 0, 0)
    optimized = KernelCapability(
        _CORTEX_M4_MAX_2X2_ID,
        200,
        True,
        _target_supported(options) and exact,
        (
            "fixed 2x2 non-overlapping window lowers to four direct loads"
            if _target_supported(options) and exact
            else "requires Cortex-M4 DSP target and 2x2 stride-two zero-padding shape"
        ),
    )
    portable = KernelCapability(
        _PORTABLE_MAX_ID,
        0,
        False,
        True,
        "portable MaxPool2D supports every verified step",
    )
    return esp_nn_pool_capability(step, plan, options), cmsis, optimized, portable


def _cmsis_pool_kernel(
    context: StepEmitContext,
    *,
    average: bool,
) -> KernelEmission:
    operation = "average_pool2d" if average else "max_pool2d"
    cmsis_function = "arm_avgpool_s8" if average else "arm_max_pool_s8"
    function = f"{context.symbol}_{operation}_cmsis_nn_s8"
    signature = f"""void {function}(
    const int8_t *input,
    int8_t *output,
    void *scratch,
    int32_t scratch_bytes,
    int32_t input_height,
    int32_t input_width,
    int32_t channels,
    int32_t output_height,
    int32_t output_width,
    int32_t kernel_height,
    int32_t kernel_width,
    int32_t stride_height,
    int32_t stride_width,
    int32_t pad_top,
    int32_t pad_left,
    int32_t activation_min,
    int32_t activation_max)"""
    return KernelEmission(
        key=f"cmsis_nn_{operation}_s8_v4_0_0",
        header_includes=("<stddef.h>", "<stdint.h>", '"arm_nnfunctions.h"'),
        declaration=signature + ";",
        definition=f"""{signature} {{
    const cmsis_nn_context cmsis_context = {{
        .buf = scratch,
        .size = scratch_bytes,
    }};
    const cmsis_nn_pool_params parameters = {{
        .stride = {{ .w = stride_width, .h = stride_height }},
        .padding = {{ .w = pad_left, .h = pad_top }},
        .activation = {{ .min = activation_min, .max = activation_max }},
    }};
    const cmsis_nn_dims input_dimensions = {{
        .n = 1, .h = input_height, .w = input_width, .c = channels,
    }};
    const cmsis_nn_dims filter_dimensions = {{
        .n = 1, .h = kernel_height, .w = kernel_width, .c = channels,
    }};
    const cmsis_nn_dims output_dimensions = {{
        .n = 1, .h = output_height, .w = output_width, .c = channels,
    }};
    (void){cmsis_function}(
        &cmsis_context, &parameters,
        &input_dimensions, input,
        &filter_dimensions, &output_dimensions, output);
}}""",
    )


def _average_kernel(context: StepEmitContext) -> KernelEmission:
    symbol = context.symbol
    function = f"{symbol}_average_pool2d_s8"
    return KernelEmission(
        key="average_pool2d_s8",
        header_includes=("<stddef.h>", "<stdint.h>"),
        declaration=f"""void {function}(
    const int8_t *input,
    int8_t *output,
    size_t input_h, size_t input_w, size_t channels,
    size_t output_h, size_t output_w,
    size_t kernel_h, size_t kernel_w,
    size_t stride_h, size_t stride_w,
    size_t pad_top, size_t pad_left,
    int32_t zero_point,
    int32_t activation_min, int32_t activation_max);""",
        definition=f"""static int32_t {symbol}_pool_round_divide_away(int32_t value, int32_t divisor) {{
    const uint32_t magnitude = value < 0 ? (uint32_t)(-(int64_t)value) : (uint32_t)value;
    const uint32_t rounded = (magnitude + (uint32_t)divisor / UINT32_C(2)) / (uint32_t)divisor;
    return value < 0 ? -(int32_t)rounded : (int32_t)rounded;
}}

void {function}(
    const int8_t *input,
    int8_t *output,
    size_t input_h, size_t input_w, size_t channels,
    size_t output_h, size_t output_w,
    size_t kernel_h, size_t kernel_w,
    size_t stride_h, size_t stride_w,
    size_t pad_top, size_t pad_left,
    int32_t zero_point,
    int32_t activation_min, int32_t activation_max) {{
    for (size_t output_y = 0; output_y < output_h; ++output_y) {{
        for (size_t output_x = 0; output_x < output_w; ++output_x) {{
            for (size_t channel = 0; channel < channels; ++channel) {{
                int32_t accumulator = 0;
                size_t valid_count = 0;
                for (size_t kernel_y = 0; kernel_y < kernel_h; ++kernel_y) {{
                    const int64_t input_y = (int64_t)output_y * (int64_t)stride_h
                        + (int64_t)kernel_y - (int64_t)pad_top;
                    if (input_y < 0 || input_y >= (int64_t)input_h) {{
                        continue;
                    }}
                    for (size_t kernel_x = 0; kernel_x < kernel_w; ++kernel_x) {{
                        const int64_t input_x = (int64_t)output_x * (int64_t)stride_w
                            + (int64_t)kernel_x - (int64_t)pad_left;
                        if (input_x < 0 || input_x >= (int64_t)input_w) {{
                            continue;
                        }}
                        const size_t input_index =
                            (((size_t)input_y * input_w + (size_t)input_x) * channels) + channel;
                        accumulator += (int32_t)input[input_index] - zero_point;
                        ++valid_count;
                    }}
                }}
                int32_t result = {symbol}_pool_round_divide_away(
                    accumulator, (int32_t)valid_count) + zero_point;
                if (result < activation_min) {{
                    result = activation_min;
                }} else if (result > activation_max) {{
                    result = activation_max;
                }}
                const size_t output_index =
                    ((output_y * output_w + output_x) * channels) + channel;
                output[output_index] = (int8_t)result;
            }}
        }}
    }}
}}""",
    )


def _max_kernel(context: StepEmitContext) -> KernelEmission:
    symbol = context.symbol
    function = f"{symbol}_max_pool2d_s8"
    return KernelEmission(
        key="max_pool2d_s8",
        header_includes=("<stddef.h>", "<stdint.h>"),
        declaration=f"""void {function}(
    const int8_t *input,
    int8_t *output,
    size_t input_h, size_t input_w, size_t channels,
    size_t output_h, size_t output_w,
    size_t kernel_h, size_t kernel_w,
    size_t stride_h, size_t stride_w,
    size_t pad_top, size_t pad_left,
    int32_t activation_min, int32_t activation_max);""",
        definition=f"""void {function}(
    const int8_t *input,
    int8_t *output,
    size_t input_h, size_t input_w, size_t channels,
    size_t output_h, size_t output_w,
    size_t kernel_h, size_t kernel_w,
    size_t stride_h, size_t stride_w,
    size_t pad_top, size_t pad_left,
    int32_t activation_min, int32_t activation_max) {{
    for (size_t output_y = 0; output_y < output_h; ++output_y) {{
        for (size_t output_x = 0; output_x < output_w; ++output_x) {{
            for (size_t channel = 0; channel < channels; ++channel) {{
                int32_t result = INT32_MIN;
                for (size_t kernel_y = 0; kernel_y < kernel_h; ++kernel_y) {{
                    const int64_t input_y = (int64_t)output_y * (int64_t)stride_h
                        + (int64_t)kernel_y - (int64_t)pad_top;
                    if (input_y < 0 || input_y >= (int64_t)input_h) {{
                        continue;
                    }}
                    for (size_t kernel_x = 0; kernel_x < kernel_w; ++kernel_x) {{
                        const int64_t input_x = (int64_t)output_x * (int64_t)stride_w
                            + (int64_t)kernel_x - (int64_t)pad_left;
                        if (input_x < 0 || input_x >= (int64_t)input_w) {{
                            continue;
                        }}
                        const size_t input_index =
                            (((size_t)input_y * input_w + (size_t)input_x) * channels) + channel;
                        const int32_t candidate = (int32_t)input[input_index];
                        if (candidate > result) {{
                            result = candidate;
                        }}
                    }}
                }}
                if (result < activation_min) {{
                    result = activation_min;
                }} else if (result > activation_max) {{
                    result = activation_max;
                }}
                const size_t output_index =
                    ((output_y * output_w + output_x) * channels) + channel;
                output[output_index] = (int8_t)result;
            }}
        }}
    }}
}}""",
        source_includes=("<limits.h>",),
    )


def _global_average_kernel(context: StepEmitContext) -> KernelEmission:
    symbol = context.symbol
    function = f"{symbol}_global_average_pool2d_s8"
    return KernelEmission(
        key="cortex_m4_global_average_pool2d_s8_v1",
        header_includes=("<stddef.h>", "<stdint.h>"),
        declaration=f"""void {function}(
    const int8_t *, int8_t *, size_t, size_t, int32_t, int32_t, int32_t);""",
        definition=f"""void {function}(
    const int8_t *input, int8_t *output, size_t positions, size_t channels,
    int32_t zero_point, int32_t activation_min, int32_t activation_max) {{
    for (size_t channel = 0; channel < channels; ++channel) {{
        int32_t accumulator = 0;
        for (size_t position = 0; position < positions; ++position) {{
            accumulator += (int32_t)input[position * channels + channel] - zero_point;
        }}
        const uint32_t magnitude = accumulator < 0
            ? (uint32_t)(-(int64_t)accumulator) : (uint32_t)accumulator;
        const uint32_t rounded =
            (magnitude + (uint32_t)positions / 2u) / (uint32_t)positions;
        int32_t result = (accumulator < 0 ? -(int32_t)rounded : (int32_t)rounded)
            + zero_point;
        if (result < activation_min) result = activation_min;
        else if (result > activation_max) result = activation_max;
        output[channel] = (int8_t)result;
    }}
}}""",
    )


def _max_2x2_kernel(context: StepEmitContext) -> KernelEmission:
    symbol = context.symbol
    function = f"{symbol}_max_pool2d_2x2_s2_s8"
    return KernelEmission(
        key="cortex_m4_max_pool2d_2x2_s2_s8_v1",
        header_includes=("<stddef.h>", "<stdint.h>"),
        declaration=f"""void {function}(
    const int8_t *, int8_t *, size_t, size_t, size_t, size_t, size_t,
    int32_t, int32_t);""",
        definition=f"""void {function}(
    const int8_t *input, int8_t *output, size_t input_width, size_t channels,
    size_t output_height, size_t output_width, size_t output_channels,
    int32_t activation_min, int32_t activation_max) {{
    (void)output_channels;
    for (size_t y = 0; y < output_height; ++y) {{
        for (size_t x = 0; x < output_width; ++x) {{
            const size_t base = ((y * 2u) * input_width + x * 2u) * channels;
            for (size_t channel = 0; channel < channels; ++channel) {{
                int32_t result = input[base + channel];
                const int32_t b = input[base + channels + channel];
                const int32_t c = input[base + input_width * channels + channel];
                const int32_t d = input[base + (input_width + 1u) * channels + channel];
                if (b > result) result = b;
                if (c > result) result = c;
                if (d > result) result = d;
                if (result < activation_min) result = activation_min;
                else if (result > activation_max) result = activation_max;
                output[(y * output_width + x) * channels + channel] = (int8_t)result;
            }}
        }}
    }}
}}""",
    )


def _call(
    step: AveragePool2DStep | MaxPool2DStep,
    context: StepEmitContext,
    function: str,
    *,
    average: bool,
) -> str:
    input_type = context.plan.tensors[step.input].tensor_type
    output_type = context.plan.tensors[step.output].tensor_type
    input_qparams = input_type.qparams
    assert isinstance(input_qparams, PerTensorQParams)
    _, input_h, input_w, channels = input_type.shape
    _, output_h, output_w, _ = output_type.shape
    kernel_h, kernel_w = step.kernel
    stride_h, stride_w = step.stride
    pad_top, _, pad_left, _ = step.padding
    qparam_line = f"        {input_qparams.zero_point},\n" if average else ""
    return (
        f"    {function}(\n"
        f"        {context.pointer(step.input, mutable=False)},\n"
        f"        {context.pointer(step.output, mutable=True)},\n"
        f"        {input_h}u, {input_w}u, {channels}u,\n"
        f"        {output_h}u, {output_w}u,\n"
        f"        {kernel_h}u, {kernel_w}u, {stride_h}u, {stride_w}u,\n"
        f"        {pad_top}u, {pad_left}u,\n"
        + qparam_line
        + f"        {step.activation_min}, {step.activation_max});"
    )


@emit_step.register
def _emit_average_pool(step: AveragePool2DStep, context: StepEmitContext) -> StepEmission:
    implementation = _PORTABLE_AVERAGE_ID if context.selection is None else context.selection.kernel_id
    if implementation in ESP_NN_AVERAGE_POOL_IDS.values():
        kernel, call = esp_nn_pool_emission(step, context)
    elif implementation == _CMSIS_NN_AVERAGE_ID:
        function = f"{context.symbol}_average_pool2d_cmsis_nn_s8"
        input_type = context.plan.tensors[step.input].tensor_type
        output_type = context.plan.tensors[step.output].tensor_type
        _, input_h, input_w, channels = input_type.shape
        _, output_h, output_w, _ = output_type.shape
        kernel_h, kernel_w = step.kernel
        stride_h, stride_w = step.stride
        pad_top, _, pad_left, _ = step.padding
        scratch_size = 0 if context.selection is None else context.selection.scratch_size
        scratch_pointer = context.scratch_pointer if scratch_size else "NULL"
        call = f"""    {function}(
        {context.pointer(step.input, mutable=False)},
        {context.pointer(step.output, mutable=True)},
        {scratch_pointer}, {scratch_size},
        {input_h}, {input_w}, {channels},
        {output_h}, {output_w},
        {kernel_h}, {kernel_w}, {stride_h}, {stride_w},
        {pad_top}, {pad_left},
        {step.activation_min}, {step.activation_max});"""
        kernel = _cmsis_pool_kernel(context, average=True)
    elif implementation == _CORTEX_M4_GLOBAL_AVERAGE_ID:
        function = f"{context.symbol}_global_average_pool2d_s8"
        input_type = context.plan.tensors[step.input].tensor_type
        qparams = input_type.qparams
        assert isinstance(qparams, PerTensorQParams)
        call = (
            f"    {function}({context.pointer(step.input, mutable=False)}, "
            f"{context.pointer(step.output, mutable=True)}, "
            f"{input_type.shape[1] * input_type.shape[2]}u, {input_type.shape[3]}u, "
            f"{qparams.zero_point}, {step.activation_min}, {step.activation_max});"
        )
        kernel = _global_average_kernel(context)
    elif implementation == _PORTABLE_AVERAGE_ID:
        function = f"{context.symbol}_average_pool2d_s8"
        call = _call(step, context, function, average=True)
        kernel = _average_kernel(context)
    else:
        raise CompileError(f"unsupported AveragePool2D implementation {implementation}")
    return StepEmission(
        constants=(),
        kernels=(kernel,),
        call=call,
        manifest={
            "name": step.name,
            "kind": step.kernel_kind,
            "input": step.input,
            "output": step.output,
            "accumulator_bound": step.accumulator_bound,
        },
    )


@emit_step.register
def _emit_max_pool(step: MaxPool2DStep, context: StepEmitContext) -> StepEmission:
    implementation = _PORTABLE_MAX_ID if context.selection is None else context.selection.kernel_id
    if implementation in ESP_NN_MAX_POOL_IDS.values():
        kernel, call = esp_nn_pool_emission(step, context)
    elif implementation == _CMSIS_NN_MAX_ID:
        function = f"{context.symbol}_max_pool2d_cmsis_nn_s8"
        input_type = context.plan.tensors[step.input].tensor_type
        output_type = context.plan.tensors[step.output].tensor_type
        _, input_h, input_w, channels = input_type.shape
        _, output_h, output_w, _ = output_type.shape
        kernel_h, kernel_w = step.kernel
        stride_h, stride_w = step.stride
        pad_top, _, pad_left, _ = step.padding
        call = f"""    {function}(
        {context.pointer(step.input, mutable=False)},
        {context.pointer(step.output, mutable=True)},
        NULL, 0,
        {input_h}, {input_w}, {channels},
        {output_h}, {output_w},
        {kernel_h}, {kernel_w}, {stride_h}, {stride_w},
        {pad_top}, {pad_left},
        {step.activation_min}, {step.activation_max});"""
        kernel = _cmsis_pool_kernel(context, average=False)
    elif implementation == _CORTEX_M4_MAX_2X2_ID:
        function = f"{context.symbol}_max_pool2d_2x2_s2_s8"
        input_type = context.plan.tensors[step.input].tensor_type
        output_type = context.plan.tensors[step.output].tensor_type
        call = (
            f"    {function}({context.pointer(step.input, mutable=False)}, "
            f"{context.pointer(step.output, mutable=True)}, {input_type.shape[2]}u, "
            f"{input_type.shape[3]}u, {output_type.shape[1]}u, {output_type.shape[2]}u, "
            f"{output_type.shape[3]}u, {step.activation_min}, {step.activation_max});"
        )
        kernel = _max_2x2_kernel(context)
    elif implementation == _PORTABLE_MAX_ID:
        function = f"{context.symbol}_max_pool2d_s8"
        call = _call(step, context, function, average=False)
        kernel = _max_kernel(context)
    else:
        raise CompileError(f"unsupported MaxPool2D implementation {implementation}")
    return StepEmission(
        constants=(),
        kernels=(kernel,),
        call=call,
        manifest={
            "name": step.name,
            "kind": step.kernel_kind,
            "input": step.input,
            "output": step.output,
        },
    )


__all__: list[str] = []
