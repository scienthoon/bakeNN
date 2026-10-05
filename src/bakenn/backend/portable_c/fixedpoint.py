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
    /* |INT32_MIN| fits uint32_t and exponent >= 1 keeps the quotient below
       2^31, so 32-bit arithmetic is exact for every int32 value. */
    const uint32_t magnitude =
        value < 0 ? UINT32_C(0) - (uint32_t)value : (uint32_t)value;
    const uint32_t mask = (UINT32_C(1) << (uint32_t)exponent) - UINT32_C(1);
    const uint32_t quotient = (magnitude >> (uint32_t)exponent)
        + (uint32_t)((magnitude & mask) > (mask >> 1));
    return value < 0 ? -(int32_t)quotient : (int32_t)quotient;
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
