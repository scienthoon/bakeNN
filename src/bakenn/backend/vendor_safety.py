"""Arithmetic constraints of the pinned CMSIS-NN and ESP-NN kernels."""

from bakenn.quantization.fixedpoint import (
    INT32_MAX,
    INT32_MIN,
    multiply_by_quantized_multiplier,
)


def requantization_failure(
    bounds: tuple[int, ...],
    multipliers: tuple[int, ...],
    shifts: tuple[int, ...],
    output_zero_point: int,
) -> str | None:
    """Check vendor intermediates which are wider in BakeNN's own kernels.

    Both pinned libraries construct their right-shift mask with signed
    ``1 << exponent`` and add the output zero point in int32 before clamping.
    Positive Q31 requantization is monotone, so the proven accumulator
    interval's endpoints also bound this final addition.
    """

    if any(shift < -30 or shift > 30 for shift in shifts):
        return "vendor requantization requires shifts in [-30, 30] for signed shift masks"
    for bound, multiplier, shift in zip(bounds, multipliers, shifts):
        minimum = multiply_by_quantized_multiplier(-bound, multiplier, shift)
        maximum = multiply_by_quantized_multiplier(bound, multiplier, shift)
        if (
            minimum + output_zero_point < INT32_MIN
            or maximum + output_zero_point > INT32_MAX
        ):
            return "vendor requantization output zero-point addition can overflow int32"
    return None
