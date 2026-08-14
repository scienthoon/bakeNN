from __future__ import annotations

from .contracts import KernelEmission, StepEmitContext


def q31_kernel(context: StepEmitContext) -> KernelEmission:
    """Emit the sole C implementation of ``bakenn.int8.v1`` requantization."""

    symbol = context.symbol
    high_mul = f"{symbol}_q31_high_mul"
    round_pot = f"{symbol}_q31_round_div_pot"
    requantize = f"{symbol}_q31_requantize"
    clamp = f"{symbol}_q31_clamp_s8"
    return KernelEmission(
        key="bakenn_q31_v1",
        header_includes=("<stdint.h>",),
        source_includes=("<limits.h>",),
        declaration=f"""int32_t {high_mul}(int32_t a, int32_t b);
int32_t {round_pot}(int32_t value, int32_t exponent);
int32_t {requantize}(int32_t value, int32_t multiplier, int32_t shift);
int8_t {clamp}(int64_t value, int32_t minimum, int32_t maximum);""",
        definition=f"""int32_t {high_mul}(int32_t a, int32_t b) {{
    if (a == INT32_MIN && b == INT32_MIN) {{
        return INT32_MAX;
    }}
    const int64_t product = (int64_t)a * (int64_t)b;
    const int64_t nudge =
        product >= 0 ? (INT64_C(1) << 30) : INT64_C(1) - (INT64_C(1) << 30);
    return (int32_t)((product + nudge) / (INT64_C(1) << 31));
}}

int32_t {round_pot}(int32_t value, int32_t exponent) {{
    if (exponent == 0) {{
        return value;
    }}
    const uint64_t magnitude =
        value < 0 ? (uint64_t)(-(int64_t)value) : (uint64_t)value;
    uint64_t quotient = magnitude >> (uint32_t)exponent;
    const uint64_t remainder =
        magnitude & ((UINT64_C(1) << (uint32_t)exponent) - UINT64_C(1));
    if (remainder >= (UINT64_C(1) << (uint32_t)(exponent - 1))) {{
        ++quotient;
    }}
    return value < 0 ? (int32_t)(-(int64_t)quotient) : (int32_t)quotient;
}}

int32_t {requantize}(int32_t value, int32_t multiplier, int32_t shift) {{
    const int32_t left_shift = shift > 0 ? shift : 0;
    const int32_t right_shift = shift < 0 ? -shift : 0;
    const int64_t shifted64 =
        (int64_t)value * (INT64_C(1) << (uint32_t)left_shift);
    const int32_t shifted = (int32_t)shifted64;
    return {round_pot}({high_mul}(shifted, multiplier), right_shift);
}}

int8_t {clamp}(int64_t value, int32_t minimum, int32_t maximum) {{
    if (value < minimum) {{
        value = minimum;
    }} else if (value > maximum) {{
        value = maximum;
    }}
    return (int8_t)value;
}}""",
    )


def q31_requantize_name(context: StepEmitContext) -> str:
    return f"{context.symbol}_q31_requantize"


def clamp_s8_name(context: StepEmitContext) -> str:
    return f"{context.symbol}_q31_clamp_s8"


__all__ = ["clamp_s8_name", "q31_kernel", "q31_requantize_name"]
