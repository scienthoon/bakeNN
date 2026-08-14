from __future__ import annotations

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


def _target_supported(options: CBackendOptions) -> bool:
    return "dsp" in options.target.features and "armv7e-m" in options.target.features


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
    return optimized, portable


@kernel_capabilities.register
def _max_capabilities(
    step: MaxPool2DStep,
    plan: ExecutionPlan,
    options: CBackendOptions,
) -> tuple[KernelCapability, ...]:
    del plan
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
    return optimized, portable


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
    if implementation == _CORTEX_M4_GLOBAL_AVERAGE_ID:
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
    if implementation == _CORTEX_M4_MAX_2X2_ID:
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
