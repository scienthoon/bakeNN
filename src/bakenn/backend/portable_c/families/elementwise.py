from __future__ import annotations

from bakenn.ir.types import PerTensorQParams
from bakenn.plan.steps.elementwise import AddStep, ClampStep, MulStep, RequantizeStep

from ..contracts import KernelEmission, StepEmitContext, StepEmission, emit_step
from ..fixedpoint import clamp_s8_name, q31_kernel, q31_requantize_name


def _kernel(context: StepEmitContext) -> KernelEmission:
    symbol = context.symbol
    requantize = q31_requantize_name(context)
    clamp = clamp_s8_name(context)
    return KernelEmission(
        key="elementwise_s8",
        header_includes=("<stddef.h>", "<stdint.h>"),
        source_includes=(),
        declaration=f"""void {symbol}_add_s8(
    const int8_t *input_a,
    const int8_t *input_b,
    int8_t *output,
    size_t dim0, size_t dim1, size_t dim2, size_t dim3,
    size_t a_stride0, size_t a_stride1, size_t a_stride2, size_t a_stride3,
    size_t b_stride0, size_t b_stride1, size_t b_stride2, size_t b_stride3,
    int32_t input_a_zero_point,
    int32_t input_b_zero_point,
    int32_t output_zero_point,
    int32_t input_a_multiplier,
    int32_t input_a_shift,
    int32_t input_b_multiplier,
    int32_t input_b_shift,
    int32_t output_multiplier,
    int32_t output_shift,
    int32_t activation_min,
    int32_t activation_max);

void {symbol}_mul_s8(
    const int8_t *input_a,
    const int8_t *input_b,
    int8_t *output,
    size_t dim0, size_t dim1, size_t dim2, size_t dim3,
    size_t a_stride0, size_t a_stride1, size_t a_stride2, size_t a_stride3,
    size_t b_stride0, size_t b_stride1, size_t b_stride2, size_t b_stride3,
    int32_t input_a_zero_point,
    int32_t input_b_zero_point,
    int32_t output_zero_point,
    int32_t output_multiplier,
    int32_t output_shift,
    int32_t activation_min,
    int32_t activation_max);

void {symbol}_clamp_s8(
    const int8_t *input,
    int8_t *output,
    size_t count,
    int32_t activation_min,
    int32_t activation_max);

void {symbol}_requantize_s8(
    const int8_t *input,
    int8_t *output,
    size_t count,
    int32_t input_zero_point,
    int32_t output_zero_point,
    int32_t multiplier,
    int32_t shift,
    int32_t activation_min,
    int32_t activation_max);""",
        definition=f"""void {symbol}_add_s8(
    const int8_t *input_a,
    const int8_t *input_b,
    int8_t *output,
    size_t dim0, size_t dim1, size_t dim2, size_t dim3,
    size_t a_stride0, size_t a_stride1, size_t a_stride2, size_t a_stride3,
    size_t b_stride0, size_t b_stride1, size_t b_stride2, size_t b_stride3,
    int32_t input_a_zero_point,
    int32_t input_b_zero_point,
    int32_t output_zero_point,
    int32_t input_a_multiplier,
    int32_t input_a_shift,
    int32_t input_b_multiplier,
    int32_t input_b_shift,
    int32_t output_multiplier,
    int32_t output_shift,
    int32_t activation_min,
    int32_t activation_max) {{
    size_t output_index = 0;
    for (size_t i0 = 0; i0 < dim0; ++i0) {{
    for (size_t i1 = 0; i1 < dim1; ++i1) {{
    for (size_t i2 = 0; i2 < dim2; ++i2) {{
    for (size_t i3 = 0; i3 < dim3; ++i3) {{
        const size_t a_index = i0 * a_stride0 + i1 * a_stride1 + i2 * a_stride2 + i3 * a_stride3;
        const size_t b_index = i0 * b_stride0 + i1 * b_stride1 + i2 * b_stride2 + i3 * b_stride3;
        const int32_t centered_a = (int32_t)input_a[a_index] - input_a_zero_point;
        const int32_t centered_b = (int32_t)input_b[b_index] - input_b_zero_point;
        const int32_t shifted_a = (int32_t)((int64_t)centered_a * (INT64_C(1) << 20));
        const int32_t shifted_b = (int32_t)((int64_t)centered_b * (INT64_C(1) << 20));
        const int32_t scaled_a = {requantize}(
            shifted_a, input_a_multiplier, input_a_shift);
        const int32_t scaled_b = {requantize}(
            shifted_b, input_b_multiplier, input_b_shift);
        const int32_t sum = (int32_t)((int64_t)scaled_a + (int64_t)scaled_b);
        const int32_t scaled_sum = {requantize}(
            sum, output_multiplier, output_shift);
        output[output_index++] = {clamp}(
            (int64_t)scaled_sum + output_zero_point, activation_min, activation_max);
    }}}}}}}}
}}

void {symbol}_mul_s8(
    const int8_t *input_a,
    const int8_t *input_b,
    int8_t *output,
    size_t dim0, size_t dim1, size_t dim2, size_t dim3,
    size_t a_stride0, size_t a_stride1, size_t a_stride2, size_t a_stride3,
    size_t b_stride0, size_t b_stride1, size_t b_stride2, size_t b_stride3,
    int32_t input_a_zero_point,
    int32_t input_b_zero_point,
    int32_t output_zero_point,
    int32_t output_multiplier,
    int32_t output_shift,
    int32_t activation_min,
    int32_t activation_max) {{
    size_t output_index = 0;
    for (size_t i0 = 0; i0 < dim0; ++i0) {{
    for (size_t i1 = 0; i1 < dim1; ++i1) {{
    for (size_t i2 = 0; i2 < dim2; ++i2) {{
    for (size_t i3 = 0; i3 < dim3; ++i3) {{
        const size_t a_index = i0 * a_stride0 + i1 * a_stride1 + i2 * a_stride2 + i3 * a_stride3;
        const size_t b_index = i0 * b_stride0 + i1 * b_stride1 + i2 * b_stride2 + i3 * b_stride3;
        const int32_t centered_a = (int32_t)input_a[a_index] - input_a_zero_point;
        const int32_t centered_b = (int32_t)input_b[b_index] - input_b_zero_point;
        const int32_t product = (int32_t)((int64_t)centered_a * (int64_t)centered_b);
        const int32_t scaled = {requantize}(
            product, output_multiplier, output_shift);
        output[output_index++] = {clamp}(
            (int64_t)scaled + output_zero_point, activation_min, activation_max);
    }}}}}}}}
}}

void {symbol}_clamp_s8(
    const int8_t *input,
    int8_t *output,
    size_t count,
    int32_t activation_min,
    int32_t activation_max) {{
    for (size_t index = 0; index < count; ++index) {{
        output[index] = {clamp}(
            input[index], activation_min, activation_max);
    }}
}}

void {symbol}_requantize_s8(
    const int8_t *input,
    int8_t *output,
    size_t count,
    int32_t input_zero_point,
    int32_t output_zero_point,
    int32_t multiplier,
    int32_t shift,
    int32_t activation_min,
    int32_t activation_max) {{
    for (size_t index = 0; index < count; ++index) {{
        const int32_t centered = (int32_t)input[index] - input_zero_point;
        const int32_t scaled = {requantize}(centered, multiplier, shift);
        output[index] = {clamp}(
            (int64_t)scaled + output_zero_point, activation_min, activation_max);
    }}
}}""",
    )


def _qparams(context: StepEmitContext, value: str) -> PerTensorQParams:
    qparams = context.plan.tensors[value].tensor_type.qparams
    assert isinstance(qparams, PerTensorQParams)
    return qparams


def _shape4(context: StepEmitContext, value: str) -> tuple[int, int, int, int]:
    shape = context.plan.tensors[value].tensor_type.shape
    return (1,) * (4 - len(shape)) + shape  # type: ignore[return-value]


def _broadcast_strides(
    input_shape: tuple[int, int, int, int],
    output_shape: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    contiguous = [0, 0, 0, 1]
    for axis in range(2, -1, -1):
        contiguous[axis] = contiguous[axis + 1] * input_shape[axis + 1]
    return tuple(
        0 if input_shape[axis] == 1 and output_shape[axis] != 1 else contiguous[axis]
        for axis in range(4)
    )  # type: ignore[return-value]


def _broadcast_arguments(
    step: AddStep | MulStep, context: StepEmitContext
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int]]:
    output_shape = _shape4(context, step.output)
    return (
        output_shape,
        _broadcast_strides(_shape4(context, step.input_a), output_shape),
        _broadcast_strides(_shape4(context, step.input_b), output_shape),
    )


@emit_step.register
def _emit_add(step: AddStep, context: StepEmitContext) -> StepEmission:
    input_a_qparams = _qparams(context, step.input_a)
    input_b_qparams = _qparams(context, step.input_b)
    output_qparams = _qparams(context, step.output)
    output_shape, a_strides, b_strides = _broadcast_arguments(step, context)
    kernel_fn = f"{context.symbol}_add_s8"
    return StepEmission(
        constants=(),
        kernels=(q31_kernel(context), _kernel(context)),
        call=(
            f"    {kernel_fn}(\n"
            f"        {context.pointer(step.input_a, mutable=False)},\n"
            f"        {context.pointer(step.input_b, mutable=False)},\n"
            f"        {context.pointer(step.output, mutable=True)},\n"
            f"        {', '.join(f'{value}u' for value in output_shape)},\n"
            f"        {', '.join(f'{value}u' for value in a_strides)},\n"
            f"        {', '.join(f'{value}u' for value in b_strides)},\n"
            f"        {input_a_qparams.zero_point}, {input_b_qparams.zero_point}, "
            f"{output_qparams.zero_point},\n"
            f"        {step.input_a_multiplier}, {step.input_a_shift}, "
            f"{step.input_b_multiplier}, {step.input_b_shift},\n"
            f"        {step.output_multiplier}, {step.output_shift}, "
            f"{step.activation_min}, {step.activation_max});"
        ),
        manifest={
            "name": step.name,
            "kind": step.kernel_kind,
            "left_shift": step.left_shift,
            "sum_bound": step.sum_bound,
            "output_pre_high_mul_bound": step.output_pre_high_mul_bound,
            "broadcast": _shape4(context, step.input_a) != output_shape
            or _shape4(context, step.input_b) != output_shape,
        },
    )


@emit_step.register
def _emit_mul(step: MulStep, context: StepEmitContext) -> StepEmission:
    input_a_qparams = _qparams(context, step.input_a)
    input_b_qparams = _qparams(context, step.input_b)
    output_qparams = _qparams(context, step.output)
    output_shape, a_strides, b_strides = _broadcast_arguments(step, context)
    kernel_fn = f"{context.symbol}_mul_s8"
    return StepEmission(
        constants=(),
        kernels=(q31_kernel(context), _kernel(context)),
        call=(
            f"    {kernel_fn}(\n"
            f"        {context.pointer(step.input_a, mutable=False)},\n"
            f"        {context.pointer(step.input_b, mutable=False)},\n"
            f"        {context.pointer(step.output, mutable=True)},\n"
            f"        {', '.join(f'{value}u' for value in output_shape)},\n"
            f"        {', '.join(f'{value}u' for value in a_strides)},\n"
            f"        {', '.join(f'{value}u' for value in b_strides)},\n"
            f"        {input_a_qparams.zero_point}, {input_b_qparams.zero_point}, "
            f"{output_qparams.zero_point},\n"
            f"        {step.output_multiplier}, {step.output_shift}, "
            f"{step.activation_min}, {step.activation_max});"
        ),
        manifest={
            "name": step.name,
            "kind": step.kernel_kind,
            "product_bound": step.product_bound,
            "requantize_pre_high_mul_bound": step.requantize_pre_high_mul_bound,
            "broadcast": _shape4(context, step.input_a) != output_shape
            or _shape4(context, step.input_b) != output_shape,
        },
    )


@emit_step.register
def _emit_clamp(step: ClampStep, context: StepEmitContext) -> StepEmission:
    count = context.plan.tensors[step.output].tensor_type.numel
    kernel_fn = f"{context.symbol}_clamp_s8"
    return StepEmission(
        constants=(),
        kernels=(q31_kernel(context), _kernel(context)),
        call=(
            f"    {kernel_fn}({context.pointer(step.input, mutable=False)}, "
            f"{context.pointer(step.output, mutable=True)}, {count}u, "
            f"{step.activation_min}, {step.activation_max});"
        ),
        manifest={
            "name": step.name,
            "kind": step.kernel_kind,
            "activation_min": step.activation_min,
            "activation_max": step.activation_max,
            "inplace": step.inplace,
        },
    )


@emit_step.register
def _emit_requantize(step: RequantizeStep, context: StepEmitContext) -> StepEmission:
    input_qparams = _qparams(context, step.input)
    output_qparams = _qparams(context, step.output)
    count = context.plan.tensors[step.output].tensor_type.numel
    kernel_fn = f"{context.symbol}_requantize_s8"
    return StepEmission(
        constants=(),
        kernels=(q31_kernel(context), _kernel(context)),
        call=(
            f"    {kernel_fn}({context.pointer(step.input, mutable=False)}, "
            f"{context.pointer(step.output, mutable=True)}, {count}u, "
            f"{input_qparams.zero_point}, {output_qparams.zero_point}, "
            f"{step.multiplier}, {step.shift}, "
            f"{step.activation_min}, {step.activation_max});"
        ),
        manifest={
            "name": step.name,
            "kind": step.kernel_kind,
            "requantize_pre_high_mul_bound": step.requantize_pre_high_mul_bound,
            "inplace": step.inplace,
            "activation_min": step.activation_min,
            "activation_max": step.activation_max,
        },
    )


__all__: list[str] = []
