from __future__ import annotations

import numpy as np

from bakenn.ir.types import PerTensorQParams
from bakenn.plan.steps.sequence import AveragePool1DStep, Conv1DStep, MaxPool1DStep

from ..contracts import ConstantEmission, KernelEmission, StepEmitContext, StepEmission, emit_step
from ..fixedpoint import clamp_s8_name, q31_kernel, q31_requantize_name
from ..formatting import format_values


def _int32_constant(symbol: str, values: tuple[int, ...]) -> ConstantEmission:
    array = np.asarray(values, dtype=np.int32)
    return ConstantEmission(symbol, f"extern const int32_t {symbol}[{len(values)}];", f"const int32_t {symbol}[{len(values)}] = {{\n{format_values(array)}\n}};", int(array.nbytes))


def _conv_kernel(context: StepEmitContext) -> KernelEmission:
    fn = f"{context.symbol}_conv1d_s8"
    requantize, clamp = q31_requantize_name(context), clamp_s8_name(context)
    signature = f"""void {fn}(
    const int8_t *input, const int8_t *weight, const int32_t *bias,
    const int32_t *multiplier, const int32_t *shift, int8_t *output,
    size_t input_length, size_t input_channels, size_t output_length,
    size_t output_channels, size_t kernel, size_t stride, size_t dilation,
    size_t pad_left, size_t groups, int32_t input_zero_point,
    int32_t output_zero_point, int32_t activation_min, int32_t activation_max)"""
    return KernelEmission(
        key="conv1d_s8_v1", header_includes=("<stddef.h>", "<stdint.h>"), declaration=signature + ";",
        definition=f"""{signature} {{
    const size_t group_input_channels = input_channels / groups;
    const size_t group_output_channels = output_channels / groups;
    for (size_t position = 0; position < output_length; ++position) {{
        for (size_t output_channel = 0; output_channel < output_channels; ++output_channel) {{
            const size_t group = output_channel / group_output_channels;
            const size_t input_base = group * group_input_channels;
            int32_t accumulator = bias[output_channel];
            for (size_t kernel_index = 0; kernel_index < kernel; ++kernel_index) {{
                const int64_t input_position = (int64_t)position * (int64_t)stride
                    + (int64_t)kernel_index * (int64_t)dilation - (int64_t)pad_left;
                for (size_t local_channel = 0; local_channel < group_input_channels; ++local_channel) {{
                    int32_t code = input_zero_point;
                    if (input_position >= 0 && (uint64_t)input_position < (uint64_t)input_length) {{
                        code = input[((size_t)input_position * input_channels) + input_base + local_channel];
                    }}
                    const size_t weight_index =
                        (output_channel * kernel + kernel_index) * group_input_channels + local_channel;
                    accumulator += (code - input_zero_point) * (int32_t)weight[weight_index];
                }}
            }}
            const int32_t scaled = {requantize}(accumulator, multiplier[output_channel], shift[output_channel]);
            output[position * output_channels + output_channel] = {clamp}(
                (int64_t)scaled + output_zero_point, activation_min, activation_max);
        }}
    }}
}}""",
    )


@emit_step.register
def _emit_conv1d(step: Conv1DStep, context: StepEmitContext) -> StepEmission:
    input_type, output_type, weight_type = (context.plan.tensors[name].tensor_type for name in (step.input, step.output, step.weight))
    input_qparams, output_qparams = input_type.qparams, output_type.qparams
    assert isinstance(input_qparams, PerTensorQParams) and isinstance(output_qparams, PerTensorQParams)
    multiplier = f"{context.symbol}_op{context.step_index}_multiplier"
    shift = f"{context.symbol}_op{context.step_index}_shift"
    fn = f"{context.symbol}_conv1d_s8"
    return StepEmission(
        constants=(_int32_constant(multiplier, step.multipliers), _int32_constant(shift, step.shifts)),
        kernels=(q31_kernel(context), _conv_kernel(context)),
        call=f"""    {fn}(
        {context.pointer(step.input, mutable=False)}, {context.pointer(step.weight, mutable=False)},
        {context.pointer(step.bias, mutable=False)}, {multiplier}, {shift},
        {context.pointer(step.output, mutable=True)}, {input_type.shape[1]}u, {input_type.shape[2]}u,
        {output_type.shape[1]}u, {output_type.shape[2]}u, {weight_type.shape[1]}u,
        {step.stride}u, {step.dilation}u, {step.padding[0]}u, {step.groups}u,
        {input_qparams.zero_point}, {output_qparams.zero_point}, {step.activation_min}, {step.activation_max});""",
        manifest={"name": step.name, "kind": step.kernel_kind, "groups": step.groups, "accumulator_bound_max": max(step.accumulator_bounds)},
    )


def _pool_kernel(context: StepEmitContext, *, average: bool) -> KernelEmission:
    kind = "average" if average else "max"
    fn = f"{context.symbol}_{kind}_pool1d_s8"
    helper = f"""static int32_t {context.symbol}_pool1d_round_away(int32_t value, int32_t divisor) {{
    const uint32_t magnitude = value < 0 ? (uint32_t)(-(int64_t)value) : (uint32_t)value;
    const uint32_t rounded = (magnitude + (uint32_t)divisor / 2u) / (uint32_t)divisor;
    return value < 0 ? -(int32_t)rounded : (int32_t)rounded;
}}

""" if average else ""
    signature = f"""void {fn}(
    const int8_t *input, int8_t *output, size_t input_length, size_t channels,
    size_t output_length, size_t kernel, size_t stride, size_t pad_left,
    int32_t zero_point, int32_t activation_min, int32_t activation_max)"""
    initial = "0" if average else "INT32_MIN"
    update = "accumulator += code - zero_point;" if average else "if (code > accumulator) { accumulator = code; }"
    finish = f"{context.symbol}_pool1d_round_away(accumulator, (int32_t)valid_count) + zero_point" if average else "accumulator"
    unused = "" if average else "    (void)zero_point;\n"
    unused_count = "" if average else "            (void)valid_count;\n"
    return KernelEmission(
        key=f"{kind}_pool1d_s8_v1", header_includes=("<stddef.h>", "<stdint.h>"), source_includes=(("<limits.h>",) if not average else ()), declaration=signature + ";",
        definition=f"""{helper}{signature} {{
{unused}
    for (size_t position = 0; position < output_length; ++position) {{
        for (size_t channel = 0; channel < channels; ++channel) {{
            int32_t accumulator = {initial};
            size_t valid_count = 0;
            for (size_t offset = 0; offset < kernel; ++offset) {{
                const int64_t input_position = (int64_t)position * (int64_t)stride
                    + (int64_t)offset - (int64_t)pad_left;
                if (input_position < 0 || (uint64_t)input_position >= (uint64_t)input_length) continue;
                const int32_t code = input[(size_t)input_position * channels + channel];
                {update}
                ++valid_count;
            }}
{unused_count}            int32_t result = {finish};
            if (result < activation_min) result = activation_min;
            else if (result > activation_max) result = activation_max;
            output[position * channels + channel] = (int8_t)result;
        }}
    }}
}}""",
    )


def _emit_pool(step: AveragePool1DStep | MaxPool1DStep, context: StepEmitContext, *, average: bool) -> StepEmission:
    input_type, output_type = (context.plan.tensors[name].tensor_type for name in (step.input, step.output))
    qparams = input_type.qparams
    assert isinstance(qparams, PerTensorQParams)
    kind = "average" if average else "max"
    fn = f"{context.symbol}_{kind}_pool1d_s8"
    return StepEmission(constants=(), kernels=(_pool_kernel(context, average=average),), call=f"""    {fn}(
        {context.pointer(step.input, mutable=False)}, {context.pointer(step.output, mutable=True)},
        {input_type.shape[1]}u, {input_type.shape[2]}u, {output_type.shape[1]}u,
        {step.kernel}u, {step.stride}u, {step.padding[0]}u, {qparams.zero_point},
        {step.activation_min}, {step.activation_max});""", manifest={"name": step.name, "kind": step.kernel_kind})


@emit_step.register
def _emit_average_pool1d(step: AveragePool1DStep, context: StepEmitContext) -> StepEmission:
    return _emit_pool(step, context, average=True)


@emit_step.register
def _emit_max_pool1d(step: MaxPool1DStep, context: StepEmitContext) -> StepEmission:
    return _emit_pool(step, context, average=False)


__all__: list[str] = []
