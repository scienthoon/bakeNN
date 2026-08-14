import unittest

from bakenn.quantization.fixedpoint import (
    multiply_by_quantized_multiplier,
    quantize_multiplier,
    round_half_away_from_zero,
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


if __name__ == "__main__":
    unittest.main()
