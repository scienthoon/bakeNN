from __future__ import annotations

import numpy as np

from bakenn.plan.steps.softmax import SoftmaxStep

from ..contracts import ConstantEmission, KernelEmission, StepEmitContext, StepEmission, emit_step
from ..formatting import format_values


def _lut_constant(symbol: str, values: tuple[int, ...]) -> ConstantEmission:
    array = np.asarray(values, dtype=np.uint16)
    return ConstantEmission(
        symbol=symbol,
        declaration=f"extern const uint16_t {symbol}[256];",
        definition=f"const uint16_t {symbol}[256] = {{\n{format_values(array)}\n}};",
        size_bytes=int(array.nbytes),
    )


def _kernel(context: StepEmitContext) -> KernelEmission:
    function = f"{context.symbol}_softmax_s8_q15"
    return KernelEmission(
        key="softmax_s8_q15",
        header_includes=("<stddef.h>", "<stdint.h>"),
        declaration=f"""void {function}(
    const int8_t *input,
    const uint16_t lut[256],
    int8_t *output,
    size_t row_count,
    size_t class_count);""",
        definition=f"""void {function}(
    const int8_t *input,
    const uint16_t lut[256],
    int8_t *output,
    size_t row_count,
    size_t class_count) {{
    for (size_t row = 0; row < row_count; ++row) {{
        const size_t row_base = row * class_count;
        int32_t maximum = -128;
        for (size_t channel = 0; channel < class_count; ++channel) {{
            const int32_t value = (int32_t)input[row_base + channel];
            if (value > maximum) {{
                maximum = value;
            }}
        }}
        uint32_t sum = 0;
        for (size_t channel = 0; channel < class_count; ++channel) {{
            const uint32_t difference =
                (uint32_t)(maximum - (int32_t)input[row_base + channel]);
            sum += (uint32_t)lut[difference];
        }}
        for (size_t channel = 0; channel < class_count; ++channel) {{
            const uint32_t difference =
                (uint32_t)(maximum - (int32_t)input[row_base + channel]);
            const uint64_t numerator = (uint64_t)lut[difference] * UINT64_C(256);
            uint64_t probability = (numerator + (uint64_t)sum / UINT64_C(2)) / (uint64_t)sum;
            if (probability > UINT64_C(255)) {{
                probability = UINT64_C(255);
            }}
            output[row_base + channel] = (int8_t)((int32_t)probability - 128);
        }}
    }}
}}""",
    )


@emit_step.register
def _emit_softmax(step: SoftmaxStep, context: StepEmitContext) -> StepEmission:
    lut_symbol = f"{context.symbol}_op{context.step_index}_softmax_lut_q15"
    function = f"{context.symbol}_softmax_s8_q15"
    return StepEmission(
        constants=(_lut_constant(lut_symbol, step.lut),),
        kernels=(_kernel(context),),
        call=(
            f"    {function}(\n"
            f"        {context.pointer(step.input, mutable=False)}, {lut_symbol},\n"
            f"        {context.pointer(step.output, mutable=True)},\n"
            f"        {step.row_count}u, {step.class_count}u);"
        ),
        manifest={
            "name": step.name,
            "kind": step.kernel_kind,
            "input": step.input,
            "output": step.output,
            "arithmetic_profile": step.arithmetic_profile,
            "lut_entries": len(step.lut),
            "lut_sum_bound": step.sum_bound,
        },
    )


__all__: list[str] = []
