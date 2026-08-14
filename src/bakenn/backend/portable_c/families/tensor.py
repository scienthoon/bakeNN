from __future__ import annotations

from bakenn.ir.types import PerTensorQParams
from bakenn.plan.steps.tensor import Pad2DStep, ReduceMeanStep

from ..contracts import KernelEmission, StepEmitContext, StepEmission, emit_step
from ..fixedpoint import clamp_s8_name, q31_kernel, q31_requantize_name


def _pad_kernel(context: StepEmitContext) -> KernelEmission:
    function = f"{context.symbol}_pad2d_s8"
    return KernelEmission(
        key="pad2d_s8_v1",
        header_includes=("<stddef.h>", "<stdint.h>"),
        declaration=f"""void {function}(
    const int8_t *, int8_t *, size_t, size_t, size_t, size_t, size_t, size_t,
    size_t, int32_t);""",
        definition=f"""void {function}(
    const int8_t *input, int8_t *output,
    size_t input_height, size_t input_width, size_t channels,
    size_t output_height, size_t output_width, size_t pad_top, size_t pad_left,
    int32_t zero_point) {{
    for (size_t output_y = 0; output_y < output_height; ++output_y) {{
        for (size_t output_x = 0; output_x < output_width; ++output_x) {{
            for (size_t channel = 0; channel < channels; ++channel) {{
                const size_t output_index =
                    (output_y * output_width + output_x) * channels + channel;
                if (output_y >= pad_top && output_y - pad_top < input_height
                    && output_x >= pad_left && output_x - pad_left < input_width) {{
                    const size_t input_index =
                        ((output_y - pad_top) * input_width + (output_x - pad_left))
                        * channels + channel;
                    output[output_index] = input[input_index];
                }} else {{
                    output[output_index] = (int8_t)zero_point;
                }}
            }}
        }}
    }}
}}""",
    )


@emit_step.register
def _emit_pad(step: Pad2DStep, context: StepEmitContext) -> StepEmission:
    input_type = context.plan.tensors[step.input].tensor_type
    output_type = context.plan.tensors[step.output].tensor_type
    qparams = input_type.qparams
    assert isinstance(qparams, PerTensorQParams)
    top, _, left, _ = step.padding
    function = f"{context.symbol}_pad2d_s8"
    kernel = _pad_kernel(context)
    return StepEmission(
        constants=(),
        kernels=(kernel,),
        call=(
            f"    {function}({context.pointer(step.input, mutable=False)}, "
            f"{context.pointer(step.output, mutable=True)}, {input_type.shape[1]}u, "
            f"{input_type.shape[2]}u, {input_type.shape[3]}u, {output_type.shape[1]}u, "
            f"{output_type.shape[2]}u, {top}u, {left}u, {qparams.zero_point});"
        ),
        manifest={"name": step.name, "kind": step.kernel_kind, "padding": list(step.padding)},
    )


def _mean_kernel(context: StepEmitContext) -> KernelEmission:
    function = f"{context.symbol}_reduce_mean_s8"
    requantize = q31_requantize_name(context)
    clamp = clamp_s8_name(context)
    return KernelEmission(
        key="reduce_mean_s8_v1",
        header_includes=("<stddef.h>", "<stdint.h>"),
        declaration=f"""void {function}(
    const int8_t *, int8_t *, size_t, size_t, int32_t, int32_t, int32_t, int32_t);""",
        definition=f"""void {function}(
    const int8_t *input, int8_t *output, size_t positions, size_t channels,
    int32_t input_zero_point, int32_t output_zero_point,
    int32_t multiplier, int32_t shift) {{
    for (size_t channel = 0; channel < channels; ++channel) {{
        int32_t sum = 0;
        for (size_t position = 0; position < positions; ++position) {{
            sum += (int32_t)input[position * channels + channel] - input_zero_point;
        }}
        const int32_t scaled = {requantize}(sum, multiplier, shift);
        output[channel] = {clamp}((int64_t)scaled + output_zero_point, -128, 127);
    }}
}}""",
    )


@emit_step.register
def _emit_mean(step: ReduceMeanStep, context: StepEmitContext) -> StepEmission:
    input_qparams = context.plan.tensors[step.input].tensor_type.qparams
    output_qparams = context.plan.tensors[step.output].tensor_type.qparams
    assert isinstance(input_qparams, PerTensorQParams)
    assert isinstance(output_qparams, PerTensorQParams)
    function = f"{context.symbol}_reduce_mean_s8"
    return StepEmission(
        constants=(),
        kernels=(q31_kernel(context), _mean_kernel(context)),
        call=(
            f"    {function}({context.pointer(step.input, mutable=False)}, "
            f"{context.pointer(step.output, mutable=True)}, {step.position_count}u, "
            f"{step.channels}u, {input_qparams.zero_point}, {output_qparams.zero_point}, "
            f"{step.multiplier}, {step.shift});"
        ),
        manifest={
            "name": step.name,
            "kind": step.kernel_kind,
            "positions": step.position_count,
            "accumulator_bound": step.accumulator_bound,
        },
    )


__all__: list[str] = []
