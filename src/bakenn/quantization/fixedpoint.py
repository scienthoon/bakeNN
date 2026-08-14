from __future__ import annotations

import math

from bakenn.errors import CompileError

ARITHMETIC_PROFILE = "bakenn.int8.v1"
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def round_half_away_from_zero(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError("cannot round a non-finite value")
    return math.floor(value + 0.5) if value >= 0.0 else math.ceil(value - 0.5)


def quantize_multiplier(real_multiplier: float) -> tuple[int, int]:
    """Encode a positive scale as Q31 multiplier and base-two shift.

    A multiplier whose normalized shift is below -31 maps every possible
    int32 accumulator to zero after round-to-nearest.  Encode that exact
    constant result as ``(0, 0)`` instead of rejecting otherwise valid,
    highly attenuated channels.
    """
    if not math.isfinite(real_multiplier) or real_multiplier <= 0.0:
        raise CompileError("requantization multiplier must be finite and positive")
    significand, shift = math.frexp(real_multiplier)
    multiplier = round_half_away_from_zero(significand * (1 << 31))
    if multiplier == 1 << 31:
        multiplier //= 2
        shift += 1
    if not (1 << 30) <= multiplier <= INT32_MAX:
        raise CompileError("failed to normalize Q31 multiplier")
    if shift < -31:
        return 0, 0
    if shift > 30:
        raise CompileError(f"requantization shift {shift} is outside the portable v1 range")
    return multiplier, shift


def _trunc_div(numerator: int, denominator: int) -> int:
    magnitude = abs(numerator) // denominator
    return -magnitude if numerator < 0 else magnitude


def saturating_rounding_doubling_high_mul(a: int, b: int) -> int:
    if not INT32_MIN <= a <= INT32_MAX or not INT32_MIN <= b <= INT32_MAX:
        raise OverflowError("high-multiply inputs must fit int32")
    if a == INT32_MIN and b == INT32_MIN:
        return INT32_MAX
    product = a * b
    nudge = (1 << 30) if product >= 0 else 1 - (1 << 30)
    return _trunc_div(product + nudge, 1 << 31)


def rounding_divide_by_pot(value: int, exponent: int) -> int:
    if not 0 <= exponent <= 31:
        raise ValueError("power-of-two divisor exponent must be in [0, 31]")
    if exponent == 0:
        return value
    magnitude = abs(value)
    quotient, remainder = divmod(magnitude, 1 << exponent)
    if remainder >= 1 << (exponent - 1):
        quotient += 1
    return -quotient if value < 0 else quotient


def multiply_by_quantized_multiplier(value: int, multiplier: int, shift: int) -> int:
    if not INT32_MIN <= value <= INT32_MAX:
        raise OverflowError("accumulator must fit int32")
    left_shift = max(shift, 0)
    right_shift = max(-shift, 0)
    shifted = value * (1 << left_shift)
    if not INT32_MIN <= shifted <= INT32_MAX:
        raise OverflowError("pre-multiply left shift exceeds int32")
    high = saturating_rounding_doubling_high_mul(shifted, multiplier)
    return rounding_divide_by_pot(high, right_shift)
