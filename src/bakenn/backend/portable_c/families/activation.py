from __future__ import annotations

import numpy as np

from bakenn.plan.steps.activation import LUTActivationStep

from ..contracts import ConstantEmission, KernelEmission, StepEmitContext, StepEmission, emit_step
from ..formatting import format_values


def _kernel(context: StepEmitContext) -> KernelEmission:
    function = f"{context.symbol}_activation_lut_s8"
    return KernelEmission(
        key="activation_lut_s8_v1",
        header_includes=("<stddef.h>", "<stdint.h>"),
        declaration=f"""void {function}(
    const int8_t *input, const int8_t *lut, int8_t *output, size_t count);""",
        definition=f"""void {function}(
    const int8_t *input, const int8_t *lut, int8_t *output, size_t count) {{
    for (size_t index = 0; index < count; ++index) {{
        output[index] = lut[(size_t)((int32_t)input[index] + 128)];
    }}
}}""",
    )


@emit_step.register
def _emit_lut(step: LUTActivationStep, context: StepEmitContext) -> StepEmission:
    function = f"{context.symbol}_activation_lut_s8"
    symbol = f"{context.symbol}_op{context.step_index}_{step.operation}_lut"
    array = np.asarray(step.lut, dtype=np.int8)
    constant = ConstantEmission(
        symbol=symbol,
        declaration=f"extern const int8_t {symbol}[256];",
        definition=f"const int8_t {symbol}[256] = {{\n{format_values(array)}\n}};",
        size_bytes=256,
    )
    count = context.plan.tensors[step.output].tensor_type.numel
    return StepEmission(
        constants=(constant,),
        kernels=(_kernel(context),),
        call=(
            f"    {function}({context.pointer(step.input, mutable=False)}, {symbol}, "
            f"{context.pointer(step.output, mutable=True)}, {count}u);"
        ),
        manifest={
            "name": step.name,
            "kind": step.kernel_kind,
            "operation": step.operation,
            "input": step.input,
            "output": step.output,
            "lut_bytes": 256,
        },
    )


__all__: list[str] = []
