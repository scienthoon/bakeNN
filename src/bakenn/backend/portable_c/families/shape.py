from __future__ import annotations

from bakenn.plan.steps.shape import ConcatenateStep, FlattenStep, ReshapeStep, SliceStep

from ..contracts import KernelEmission, StepEmitContext, StepEmission, emit_step


def _view_copy_kernel(context: StepEmitContext) -> KernelEmission:
    function = f"{context.symbol}_view_copy_s8"
    return KernelEmission(
        key="view_copy_s8_v1",
        header_includes=("<stddef.h>", "<stdint.h>"),
        declaration=f"void {function}(const int8_t *, int8_t *, size_t);",
        definition=f"""void {function}(const int8_t *input, int8_t *output, size_t count) {{
    for (size_t index = 0; index < count; ++index) {{
        output[index] = input[index];
    }}
}}""",
    )


def _view_emission(
    step: ReshapeStep | FlattenStep,
    context: StepEmitContext,
) -> StepEmission:
    if step.materialize:
        function = f"{context.symbol}_view_copy_s8"
        count = context.plan.tensors[step.output].tensor_type.numel
        return StepEmission(
            constants=(),
            kernels=(_view_copy_kernel(context),),
            call=(
                f"    {function}({context.pointer(step.input, mutable=False)}, "
                f"{context.pointer(step.output, mutable=True)}, {count}u);"
            ),
            manifest={
                "name": step.name,
                "kind": step.kernel_kind,
                "input": step.input,
                "output": step.output,
                "emits_kernel": True,
                "materialized_public_view": True,
            },
        )
    return StepEmission(
        constants=(),
        kernels=(),
        call=f"    /* {step.name}: {step.kernel_kind}; storage aliases {step.input}. */",
        manifest={
            "name": step.name,
            "kind": step.kernel_kind,
            "input": step.input,
            "output": step.output,
            "emits_kernel": False,
        },
    )


@emit_step.register
def _emit_reshape(step: ReshapeStep, context: StepEmitContext) -> StepEmission:
    return _view_emission(step, context)


@emit_step.register
def _emit_flatten(step: FlattenStep, context: StepEmitContext) -> StepEmission:
    return _view_emission(step, context)


def _slice_kernel(context: StepEmitContext) -> KernelEmission:
    function = f"{context.symbol}_slice_s8"
    return KernelEmission(
        key="slice_s8_v1",
        header_includes=("<stddef.h>", "<stdint.h>"),
        declaration=f"""void {function}(
    const int8_t *input, int8_t *output,
    size_t outer_size, size_t input_axis_size, size_t output_axis_size,
    size_t inner_size, size_t start, size_t step);""",
        definition=f"""void {function}(
    const int8_t *input, int8_t *output,
    size_t outer_size, size_t input_axis_size, size_t output_axis_size,
    size_t inner_size, size_t start, size_t step) {{
    for (size_t outer = 0; outer < outer_size; ++outer) {{
        for (size_t axis = 0; axis < output_axis_size; ++axis) {{
            const size_t input_base =
                (outer * input_axis_size + start + axis * step) * inner_size;
            const size_t output_base =
                (outer * output_axis_size + axis) * inner_size;
            for (size_t inner = 0; inner < inner_size; ++inner) {{
                output[output_base + inner] = input[input_base + inner];
            }}
        }}
    }}
}}""",
    )


@emit_step.register
def _emit_slice(step: SliceStep, context: StepEmitContext) -> StepEmission:
    function = f"{context.symbol}_slice_s8"
    return StepEmission(
        constants=(),
        kernels=(_slice_kernel(context),),
        call=(
            f"    {function}({context.pointer(step.input, mutable=False)}, "
            f"{context.pointer(step.output, mutable=True)}, {step.outer_size}u, "
            f"{step.input_axis_size}u, {step.output_axis_size}u, "
            f"{step.inner_size}u, {step.start}u, {step.step}u);"
        ),
        manifest={
            "name": step.name,
            "kind": step.kernel_kind,
            "axis": step.axis,
            "start": step.start,
            "step": step.step,
            "in_place": False,
        },
    )


def _concat_kernel(context: StepEmitContext) -> KernelEmission:
    function = f"{context.symbol}_concatenate_copy_s8"
    return KernelEmission(
        key="concatenate_copy_s8",
        header_includes=("<stddef.h>", "<stdint.h>"),
        declaration=f"""void {function}(
    const int8_t *input,
    int8_t *output,
    size_t outer_size,
    size_t input_axis_size,
    size_t output_axis_size,
    size_t inner_size,
    size_t output_axis_offset);""",
        definition=f"""void {function}(
    const int8_t *input,
    int8_t *output,
    size_t outer_size,
    size_t input_axis_size,
    size_t output_axis_size,
    size_t inner_size,
    size_t output_axis_offset) {{
    const size_t copy_count = input_axis_size * inner_size;
    for (size_t outer = 0; outer < outer_size; ++outer) {{
        const size_t input_base = outer * copy_count;
        const size_t output_base =
            (outer * output_axis_size + output_axis_offset) * inner_size;
        for (size_t index = 0; index < copy_count; ++index) {{
            output[output_base + index] = input[input_base + index];
        }}
    }}
}}""",
    )


@emit_step.register
def _emit_concatenate(step: ConcatenateStep, context: StepEmitContext) -> StepEmission:
    function = f"{context.symbol}_concatenate_copy_s8"
    output_axis_size = sum(step.axis_sizes)
    calls: list[str] = []
    axis_offset = 0
    for input_name, input_axis_size in zip(step.input_names, step.axis_sizes):
        calls.append(
            f"    {function}(\n"
            f"        {context.pointer(input_name, mutable=False)},\n"
            f"        {context.pointer(step.output, mutable=True)},\n"
            f"        {step.outer_size}u, {input_axis_size}u, {output_axis_size}u,\n"
            f"        {step.inner_size}u, {axis_offset}u);"
        )
        axis_offset += input_axis_size
    return StepEmission(
        constants=(),
        kernels=(_concat_kernel(context),),
        call="\n".join(calls),
        manifest={
            "name": step.name,
            "kind": step.kernel_kind,
            "inputs": list(step.input_names),
            "output": step.output,
            "axis": step.axis,
            "in_place": False,
        },
    )


__all__: list[str] = []
