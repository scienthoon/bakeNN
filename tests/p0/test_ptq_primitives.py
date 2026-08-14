from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from bakenn.errors import CompileError
from bakenn.ir.types import Layout, PerAxisQParams, PerTensorQParams
from bakenn.quantization.observers import MinMaxObserver, observe_minmax
from bakenn.quantization.primitives import (
    quantize_bias_int32,
    quantize_compute_constants,
    quantize_weights_per_channel,
)


def test_minmax_observer_is_persistent_deterministic_and_includes_real_zero() -> None:
    first_batch = np.asarray([[-1.0, 0.25], [0.5, 1.0]], dtype=np.float64)
    empty = MinMaxObserver()
    first = empty.observe(first_batch)
    second = first.observe(np.asarray([[-0.5, 0.75]], dtype=np.float32))

    assert not empty.is_observed
    assert (first.minimum, first.maximum, first.observed_elements, first.observed_batches) == (
        -1.0,
        1.0,
        4,
        1,
    )
    assert (second.minimum, second.maximum, second.observed_elements, second.observed_batches) == (
        -1.0,
        1.0,
        6,
        2,
    )
    qparams = second.activation_qparams()
    assert qparams.scale == float(np.float32(2.0 / 255.0))
    assert qparams.zero_point == -1
    assert second == MinMaxObserver().observe(first_batch).observe([[-0.5, 0.75]])
    with pytest.raises(FrozenInstanceError):
        second.minimum = 0.0  # type: ignore[misc]

    positive = MinMaxObserver().observe([1.0, 2.0]).activation_qparams()
    assert positive.scale == float(np.float32(2.0 / 255.0))
    assert positive.zero_point == -128
    negative = MinMaxObserver().observe([-2.0, -1.0]).activation_qparams()
    assert negative.scale == float(np.float32(2.0 / 255.0))
    assert negative.zero_point == 127


def test_relu_observation_discards_negative_range_and_handles_all_zero() -> None:
    observer = MinMaxObserver().observe([-7.0, -1.0])
    assert observer.activation_qparams(relu=True).scale == 1.0
    assert observer.activation_qparams(relu=True).zero_point == 0
    mixed = MinMaxObserver().observe([-7.0, 2.0]).activation_qparams(relu=True)
    assert mixed.scale == float(np.float32(2.0 / 255.0))
    assert mixed.zero_point == -128
    all_zero = MinMaxObserver().observe(np.zeros((2, 3), dtype=np.float32)).activation_qparams()
    assert (all_zero.scale, all_zero.zero_point) == (1.0, 0)


def test_observer_uses_fp32_semantics_and_rejects_bad_calibration() -> None:
    value = np.nextafter(np.float32(1.0), np.float32(2.0), dtype=np.float32)
    observer = MinMaxObserver().observe(np.asarray([value], dtype=np.float64))
    assert observer.minimum == float(value)
    assert observer.maximum == float(value)

    with pytest.raises(CompileError, match="at least one sample"):
        MinMaxObserver().observe(np.empty((0, 3), dtype=np.float32))
    with pytest.raises(CompileError, match="NaN or infinity"):
        MinMaxObserver().observe([0.0, np.nan])
    with pytest.raises(CompileError, match="NaN or infinity"):
        MinMaxObserver().observe([np.inf])
    with pytest.raises(CompileError, match="at least one observed"):
        MinMaxObserver().activation_qparams()
    with pytest.raises(CompileError, match="at least one sample"):
        observe_minmax([])
    with pytest.raises(CompileError, match="iterable"):
        observe_minmax(3.0)


@pytest.mark.parametrize(
    "values",
    [
        np.asarray([1.0 + 2.0j], dtype=np.complex64),
        np.asarray([object()], dtype=object),
        np.asarray(["1.0"], dtype=np.str_),
        np.asarray([True], dtype=np.bool_),
    ],
    ids=("complex", "object", "string", "bool"),
)
def test_observer_rejects_non_real_numeric_dtypes_before_fp32_cast(values: np.ndarray) -> None:
    with pytest.raises(CompileError, match="real numeric dtype"):
        MinMaxObserver().observe(values)


def test_observer_uses_normalized_float32_scale_before_zero_point_rounding() -> None:
    # Raw binary64 scale puts the zero point just below 103.5, while the
    # deployable float32 scale puts it just above. Host and C must choose 104.
    observer = MinMaxObserver().observe(
        np.asarray([-1_315_095.0, 133_497.78125], dtype=np.float32)
    )
    qparams = observer.activation_qparams()
    assert qparams.scale == float(np.float32(5680.756004901961))
    assert qparams.zero_point == 104


def test_oi_weight_and_bias_hand_golden_half_away_and_immutable_snapshots() -> None:
    source_weight = np.asarray([[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    source_bias = np.asarray([0.5, 0.0], dtype=np.float64)
    result = quantize_weights_per_channel(source_weight, source_bias, layout=Layout.OI)

    np.testing.assert_array_equal(
        result.values,
        np.asarray([[-127, 0, 127], [0, 0, 0]], dtype=np.int8),
    )
    assert result.layout is Layout.OI
    assert result.qparams.axis == 0
    assert result.qparams.scales[0] == float(np.float32(1.0) / np.float32(127.0))
    assert result.qparams.scales[1] == 1.0
    assert result.qparams.zero_points == (0, 0)

    source_weight[0, 0] = 99.0
    source_bias[0] = 99.0
    assert result.values[0, 0] == -127
    with pytest.raises(ValueError):
        result.values[0, 0] = 1

    bias = quantize_bias_int32(
        np.asarray([0.5, -0.5], dtype=np.float32),
        input_scale=1.0,
        weight_qparams=PerAxisQParams((1.0, 1.0), (0, 0), 0),
    )
    np.testing.assert_array_equal(bias.values, np.asarray([1, -1], dtype=np.int32))
    assert bias.qparams.scales == (1.0, 1.0)
    with pytest.raises(ValueError):
        bias.values[0] = 2


@pytest.mark.parametrize(
    ("layout", "shape", "expected_axis"),
    [
        (Layout.OI, (3, 4), 0),
        (Layout.OHWI, (3, 2, 2, 4), 0),
        (Layout.HWO, (2, 2, 3), 2),
    ],
)
def test_supported_weight_layouts_quantize_per_output_channel(
    layout: Layout,
    shape: tuple[int, ...],
    expected_axis: int,
) -> None:
    weight = np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape)
    output_channels = shape[expected_axis]
    bias = np.zeros(output_channels, dtype=np.float32)
    first = quantize_weights_per_channel(weight, bias, layout=layout)
    second = quantize_weights_per_channel(np.array(weight, copy=True), bias, layout=layout)
    assert first.qparams.axis == expected_axis
    assert len(first.qparams.scales) == output_channels
    np.testing.assert_array_equal(first.values, second.values)
    assert first.qparams == second.qparams
    moved = np.moveaxis(first.values, expected_axis, 0).reshape(output_channels, -1)
    assert np.all(np.max(np.abs(moved.astype(np.int16)), axis=1) == 127)


def test_weight_quantization_rejects_malformed_and_constant_only_channels() -> None:
    with pytest.raises(CompileError, match="one of OI"):
        quantize_weights_per_channel([[1.0]], [0.0], layout=Layout.NC)
    with pytest.raises(CompileError, match="rank-2"):
        quantize_weights_per_channel([1.0], [0.0], layout=Layout.OI)
    with pytest.raises(CompileError, match="cannot be empty"):
        quantize_weights_per_channel(np.empty((0, 2)), np.empty((0,)), layout=Layout.OI)
    with pytest.raises(CompileError, match="NaN or infinity"):
        quantize_weights_per_channel([[np.nan]], [0.0], layout=Layout.OI)
    with pytest.raises(CompileError, match="one FP32 value"):
        quantize_weights_per_channel([[1.0], [2.0]], [0.0], layout=Layout.OI)
    with pytest.raises(CompileError, match="explicit constant-channel policy.*channels: 1"):
        quantize_weights_per_channel(
            [[1.0, -1.0], [0.0, 0.0]],
            [0.0, 0.25],
            layout=Layout.OI,
        )


def test_constant_channel_policy_selects_and_proves_exact_output_code() -> None:
    weight, bias = quantize_compute_constants(
        np.zeros((2, 3), dtype=np.float32),
        np.asarray([0.1, -1000.0], dtype=np.float32),
        layout=Layout.OI,
        input_qparams=PerTensorQParams(0.03125, 17),
        output_qparams=PerTensorQParams(0.01, -3),
    )
    np.testing.assert_array_equal(weight.values, np.zeros((2, 3), dtype=np.int8))
    # Direct output quantization gives 7 for 0.1 and saturates -1000 to -128.
    assert tuple(int(value) for value in bias.values) == (10, -125)



@pytest.mark.parametrize(
    "dtype_value",
    [
        np.asarray([[1.0 + 1.0j]], dtype=np.complex64),
        np.asarray([[object()]], dtype=object),
        np.asarray([["1.0"]], dtype=np.str_),
        np.asarray([[True]], dtype=np.bool_),
    ],
    ids=("complex", "object", "string", "bool"),
)
def test_weight_and_bias_reject_non_real_numeric_dtypes(dtype_value: np.ndarray) -> None:
    with pytest.raises(CompileError, match="weight must use a real numeric dtype"):
        quantize_weights_per_channel(dtype_value, [0.0], layout=Layout.OI)

    bad_bias = dtype_value.reshape(-1)
    with pytest.raises(CompileError, match="bias must use a real numeric dtype"):
        quantize_bias_int32(
            bad_bias,
            input_scale=1.0,
            weight_qparams=PerAxisQParams((1.0,), (0,), 0),
        )


def test_bias_quantization_rejects_invalid_scales_shapes_and_int32_overflow() -> None:
    qparams = PerAxisQParams((0.5, 0.25), (0, 0), 0)
    with pytest.raises(CompileError, match="finite and positive"):
        quantize_bias_int32([0.0, 0.0], input_scale=0.0, weight_qparams=qparams)
    with pytest.raises(CompileError, match="one FP32 value"):
        quantize_bias_int32([0.0], input_scale=1.0, weight_qparams=qparams)
    with pytest.raises(CompileError, match="NaN or infinity"):
        quantize_bias_int32([0.0, np.inf], input_scale=1.0, weight_qparams=qparams)
    with pytest.raises(CompileError, match="symmetric"):
        quantize_bias_int32(
            [0.0],
            input_scale=1.0,
            weight_qparams=PerAxisQParams((1.0,), (1,), 0),
        )
    with pytest.raises(CompileError, match="exceeds int32"):
        quantize_bias_int32(
            [np.finfo(np.float32).max],
            input_scale=1e-20,
            weight_qparams=PerAxisQParams((1.0,), (0,), 0),
        )
