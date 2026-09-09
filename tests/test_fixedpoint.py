import unittest

import numpy as np
import pytest

from bakenn.quantization.fixedpoint import (
    multiply_by_quantized_multiplier,
    quantize_multiplier,
    round_half_away_from_zero,
    rounding_divide_by_pot,
    saturating_rounding_doubling_high_mul,
)


class FixedPointTests(unittest.TestCase):
    def test_round_half_away_from_zero(self):
        self.assertEqual(round_half_away_from_zero(0.5), 1)
        self.assertEqual(round_half_away_from_zero(-0.5), -1)
        self.assertEqual(round_half_away_from_zero(1.49), 1)
        self.assertEqual(round_half_away_from_zero(-1.49), -1)

    def test_q31_golden_half(self):
        multiplier, shift = quantize_multiplier(0.5)
        self.assertEqual((multiplier, shift), (1073741824, 0))
        values = [-5, -3, -1, 0, 1, 3, 5]
        actual = [multiply_by_quantized_multiplier(value, multiplier, shift) for value in values]
        self.assertEqual(actual, [-2, -1, 0, 0, 1, 2, 3])

    def test_q31_golden_quarter(self):
        multiplier, shift = quantize_multiplier(0.25)
        values = [-7, -5, -3, -1, 1, 3, 5, 7]
        actual = [multiply_by_quantized_multiplier(value, multiplier, shift) for value in values]
        self.assertEqual(actual, [-2, -1, -1, 0, 1, 1, 2, 2])

    def test_q31_exact_underflow_is_encoded_as_zero(self):
        multiplier, shift = quantize_multiplier(2.0**-33)
        self.assertEqual((multiplier, shift), (0, 0))
        values = [-(1 << 31), -1, 0, 1, (1 << 31) - 1]
        actual = [multiply_by_quantized_multiplier(value, multiplier, shift) for value in values]
        self.assertEqual(actual, [0, 0, 0, 0, 0])

    def test_q31_smallest_nonzero_boundary_stays_normalized(self):
        self.assertEqual(quantize_multiplier(2.0**-32), (1 << 30, -31))


@pytest.mark.parametrize("scalar", [np.int32, np.int64])
def test_fixedpoint_numpy_integer_scalars_use_wide_intermediates(scalar) -> None:  # type: ignore[no-untyped-def]
    with np.errstate(over="raise"):
        assert saturating_rounding_doubling_high_mul(scalar(2147483647), scalar(1073741824)) == 1073741824
        assert multiply_by_quantized_multiplier(scalar(2147483647), scalar(1073741824), scalar(0)) == 1073741824
        assert rounding_divide_by_pot(scalar(-2147483648), scalar(1)) == -1073741824


@pytest.mark.parametrize("shift", [-32, 31, 1_000_000])
def test_multiplier_rejects_nonportable_shifts_before_shifting(shift: int) -> None:
    with pytest.raises(ValueError, match="shift"):
        multiply_by_quantized_multiplier(0, 1 << 30, shift)


@pytest.mark.parametrize("value", [True, 1.5])
def test_fixedpoint_rejects_non_integer_operands(value) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="integer"):
        multiply_by_quantized_multiplier(value, 1 << 30, 0)


if __name__ == "__main__":
    unittest.main()
