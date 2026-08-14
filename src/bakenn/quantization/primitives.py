from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from bakenn.errors import CompileError
from bakenn.ir.types import (
    Layout,
    PerAxisQParams,
    PerTensorQParams,
    normalize_scale_float32,
)
from bakenn.quantization.fixedpoint import (
    INT32_MAX,
    INT32_MIN,
    multiply_by_quantized_multiplier,
    quantize_multiplier,
    round_half_away_from_zero,
)


_WEIGHT_CONTRACTS: dict[Layout, tuple[int, int]] = {
    Layout.OI: (2, 0),
    Layout.OWI: (3, 0),
    Layout.OHWI: (4, 0),
    Layout.HWO: (3, 2),
}


def _float32_snapshot(values: object, description: str) -> np.ndarray:
    try:
        source = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as error:
        raise CompileError(f"{description} must use a real numeric dtype") from error
    if source.dtype.kind not in "iuf":
        raise CompileError(f"{description} must use a real numeric dtype")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            snapshot = np.array(source, dtype=np.float32, copy=True, order="C")
    except (TypeError, ValueError, OverflowError) as error:
        raise CompileError(f"{description} cannot be represented as FP32") from error
    if snapshot.size == 0:
        raise CompileError(f"{description} cannot be empty")
    if not np.all(np.isfinite(snapshot)):
        raise CompileError(f"{description} contains NaN or infinity in FP32 semantics")
    snapshot.setflags(write=False)
    return snapshot


def _immutable_array(
    values: np.ndarray,
    dtype: np.dtype[object],
    description: str,
) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != dtype:
        raise ValueError(f"{description} values must have dtype {dtype.name}")
    snapshot = np.array(array, copy=True, order="C")
    snapshot.setflags(write=False)
    return snapshot


def _round_away(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 0.0, np.floor(values + 0.5), np.ceil(values - 0.5))


@dataclass(frozen=True)
class QuantizedWeights:
    values: np.ndarray
    qparams: PerAxisQParams
    layout: Layout

    def __post_init__(self) -> None:
        values = _immutable_array(self.values, np.dtype(np.int8), "quantized weight")
        if not isinstance(self.layout, Layout) or self.layout not in _WEIGHT_CONTRACTS:
            raise ValueError("quantized weight layout must be OI, OWI, OHWI, or HWO")
        rank, axis = _WEIGHT_CONTRACTS[self.layout]
        if values.ndim != rank or 0 in values.shape:
            raise ValueError(f"{self.layout.value} quantized weights require rank {rank}")
        if not isinstance(self.qparams, PerAxisQParams):
            raise ValueError("quantized weights require per-axis qparams")
        if self.qparams.axis != axis or len(self.qparams.scales) != values.shape[axis]:
            raise ValueError("quantized weight qparams do not match the output-channel axis")
        if any(zero_point != 0 for zero_point in self.qparams.zero_points):
            raise ValueError("symmetric quantized weight zero_points must be zero")
        if np.any(values == -128):
            raise ValueError("symmetric quantized weights must stay in [-127, 127]")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class QuantizedBias:
    values: np.ndarray
    qparams: PerAxisQParams

    def __post_init__(self) -> None:
        values = _immutable_array(self.values, np.dtype(np.int32), "quantized bias")
        if values.ndim != 1 or values.size == 0:
            raise ValueError("quantized bias must be a non-empty vector")
        if not isinstance(self.qparams, PerAxisQParams) or self.qparams.axis != 0:
            raise ValueError("quantized bias requires per-channel qparams on axis zero")
        if len(self.qparams.scales) != values.size:
            raise ValueError("quantized bias qparams must match the bias channel count")
        if any(zero_point != 0 for zero_point in self.qparams.zero_points):
            raise ValueError("quantized bias zero_points must be zero")
        object.__setattr__(self, "values", values)


def _quantize_weights_per_channel(
    weight: object,
    bias: object,
    *,
    layout: Layout,
    constant_channel_scales: Mapping[int, Real] | None = None,
) -> QuantizedWeights:
    """Symmetrically quantize FP32 weights on the canonical output axis.

    An all-zero channel receives deterministic scale 1.0 only when its bias is
    also zero.  A caller handling a proven input-independent channel may pass
    an explicit scale.  The low-level primitive otherwise rejects a nonzero
    bias instead of silently inventing deployment arithmetic.
    """

    if not isinstance(layout, Layout) or layout not in _WEIGHT_CONTRACTS:
        raise CompileError("weight layout must be one of OI, OWI, OHWI, or HWO")
    rank, output_axis = _WEIGHT_CONTRACTS[layout]
    weight_snapshot = _float32_snapshot(weight, "weight")
    if weight_snapshot.ndim != rank or 0 in weight_snapshot.shape:
        raise CompileError(f"{layout.value} weight must be a non-empty rank-{rank} tensor")
    output_channels = weight_snapshot.shape[output_axis]
    bias_snapshot = _float32_snapshot(bias, "bias")
    if bias_snapshot.ndim != 1 or bias_snapshot.shape != (output_channels,):
        raise CompileError("bias must have one FP32 value per output channel")

    reduction_axes = tuple(axis for axis in range(rank) if axis != output_axis)
    max_abs = np.max(np.abs(weight_snapshot), axis=reduction_axes)
    all_zero = max_abs == np.float32(0.0)
    if constant_channel_scales is None:
        explicit_scales: dict[int, float] = {}
    elif not isinstance(constant_channel_scales, Mapping):
        raise CompileError("constant_channel_scales must be an integer-to-scale mapping")
    else:
        explicit_scales = {}
        for channel, scale in constant_channel_scales.items():
            if isinstance(channel, bool) or not isinstance(channel, Integral):
                raise CompileError("constant-channel scale keys must be integer channels")
            normalized_channel = int(channel)
            if not 0 <= normalized_channel < output_channels:
                raise CompileError(f"constant-channel index {normalized_channel} is out of range")
            if not bool(all_zero[normalized_channel]):
                raise CompileError(
                    f"constant-channel scale was provided for nonzero channel {normalized_channel}"
                )
            try:
                explicit_scales[normalized_channel] = normalize_scale_float32(scale)
            except ValueError as error:
                raise CompileError(
                    f"constant-channel scale for channel {normalized_channel} is not deployable"
                ) from error
    nonzero_constant_channels = np.flatnonzero(all_zero & (bias_snapshot != np.float32(0.0)))
    missing = [
        int(channel)
        for channel in nonzero_constant_channels
        if int(channel) not in explicit_scales
    ]
    if missing:
        channels = ", ".join(str(channel) for channel in missing)
        raise CompileError(
            "all-zero weight channels with nonzero bias require an explicit constant-channel policy; "
            f"unsupported channels: {channels}"
        )
    scales = np.where(
        all_zero, np.float32(1.0), max_abs / np.float32(127.0)
    ).astype(np.float32)
    for channel, scale in explicit_scales.items():
        scales[channel] = np.float32(scale)
    if not np.all(np.isfinite(scales)) or np.any(scales <= np.float32(0.0)):
        raise CompileError("weight scales cannot be represented as positive float32 values")
    broadcast_shape = [1] * rank
    broadcast_shape[output_axis] = output_channels
    quantized = _round_away(weight_snapshot / scales.reshape(broadcast_shape))
    quantized = np.clip(quantized, -127, 127).astype(np.int8)
    qparams = PerAxisQParams(
        scales=tuple(float(scale) for scale in scales),
        zero_points=(0,) * output_channels,
        axis=output_axis,
    )
    return QuantizedWeights(quantized, qparams, layout)


def quantize_weights_per_channel(
    weight: object,
    bias: object,
    *,
    layout: Layout,
) -> QuantizedWeights:
    """Quantize ordinary weights, rejecting underdetermined constant channels."""

    return _quantize_weights_per_channel(weight, bias, layout=layout)


def _quantize_bias_int32(
    bias: object,
    *,
    input_scale: float,
    weight_qparams: PerAxisQParams,
    exact_channel_codes: Mapping[int, int] | None = None,
) -> QuantizedBias:
    """Quantize FP32 bias using input_scale * weight_scale[channel]."""

    if isinstance(input_scale, bool) or not isinstance(input_scale, Real):
        raise CompileError("input_scale must be a real number")
    try:
        normalized_input_scale = normalize_scale_float32(input_scale)
    except ValueError as error:
        raise CompileError("input_scale must be finite and positive in float32") from error
    if not isinstance(weight_qparams, PerAxisQParams):
        raise CompileError("weight_qparams must be per-axis quantization parameters")
    if any(zero_point != 0 for zero_point in weight_qparams.zero_points):
        raise CompileError("bias quantization requires symmetric weight zero_points")
    bias_snapshot = _float32_snapshot(bias, "bias")
    channel_count = len(weight_qparams.scales)
    if bias_snapshot.ndim != 1 or bias_snapshot.shape != (channel_count,):
        raise CompileError("bias must have one FP32 value per weight output channel")
    raw_scales = np.asarray(weight_qparams.scales, dtype=np.float64) * normalized_input_scale
    try:
        scales = np.asarray(
            [normalize_scale_float32(float(scale)) for scale in raw_scales],
            dtype=np.float64,
        )
    except ValueError as error:
        raise CompileError("bias quantization scales must be finite positive float32 values") from error
    if exact_channel_codes is None:
        overrides: dict[int, int] = {}
    elif not isinstance(exact_channel_codes, Mapping):
        raise CompileError("exact_channel_codes must be an integer mapping")
    else:
        overrides = {}
        for channel, code in exact_channel_codes.items():
            if isinstance(channel, bool) or not isinstance(channel, Integral):
                raise CompileError("exact bias-code keys must be integer channels")
            normalized_channel = int(channel)
            if not 0 <= normalized_channel < channel_count:
                raise CompileError(f"exact bias-code channel {normalized_channel} is out of range")
            if isinstance(code, bool) or not isinstance(code, Integral):
                raise CompileError("exact bias codes must be integers")
            normalized_code = int(code)
            if not INT32_MIN <= normalized_code <= INT32_MAX:
                raise CompileError("exact bias code exceeds int32")
            overrides[normalized_channel] = normalized_code

    # Bias values have already been rounded to FP32. Division is performed in
    # float64 to avoid an additional loss before integer rounding. Explicit
    # codes are reserved for channels whose zero weights have already proven
    # the desired deployment output independently of the input.
    quantized_values: list[int] = []
    for channel, (bias_value, scale) in enumerate(zip(bias_snapshot, scales)):
        if channel in overrides:
            quantized_values.append(overrides[channel])
            continue
        scaled = float(bias_value) / float(scale)
        if not np.isfinite(scaled):
            raise CompileError("quantized bias exceeds int32")
        rounded = round_half_away_from_zero(scaled)
        if not INT32_MIN <= rounded <= INT32_MAX:
            raise CompileError("quantized bias exceeds int32")
        quantized_values.append(rounded)
    quantized = np.asarray(quantized_values, dtype=np.int32)
    qparams = PerAxisQParams(
        scales=tuple(float(scale) for scale in scales),
        zero_points=(0,) * channel_count,
        axis=0,
    )
    return QuantizedBias(quantized, qparams)


def quantize_bias_int32(
    bias: object,
    *,
    input_scale: float,
    weight_qparams: PerAxisQParams,
) -> QuantizedBias:
    """Quantize ordinary FP32 bias with no caller-supplied code overrides."""

    return _quantize_bias_int32(
        bias,
        input_scale=input_scale,
        weight_qparams=weight_qparams,
    )


def quantize_compute_constants(
    weight: object,
    bias: object,
    *,
    layout: Layout,
    input_qparams: PerTensorQParams,
    output_qparams: PerTensorQParams,
) -> tuple[QuantizedWeights, QuantizedBias]:
    """Quantize one Conv/Depthwise/Linear constant set, including constant channels.

    For an all-zero weight channel with nonzero FP32 bias, the weight scale is
    otherwise mathematically underdetermined. P0 selects ``output_scale /
    input_scale`` and stores the exact centered output code as the int32 bias.
    The helper then executes the versioned Q31 arithmetic on the host and
    rejects the model unless that code is reproduced exactly. This preserves
    the quantized output semantics without claiming that the compute channel
    was removed from the generated kernel.
    """

    if not isinstance(input_qparams, PerTensorQParams) or not isinstance(
        output_qparams, PerTensorQParams
    ):
        raise CompileError("compute-constant quantization requires per-tensor activation qparams")
    if not isinstance(layout, Layout) or layout not in _WEIGHT_CONTRACTS:
        raise CompileError("weight layout must be one of OI, OHWI, or HWO")
    rank, output_axis = _WEIGHT_CONTRACTS[layout]
    weight_snapshot = _float32_snapshot(weight, "weight")
    bias_snapshot = _float32_snapshot(bias, "bias")
    if weight_snapshot.ndim != rank or 0 in weight_snapshot.shape:
        raise CompileError(f"{layout.value} weight must be a non-empty rank-{rank} tensor")
    output_channels = weight_snapshot.shape[output_axis]
    if bias_snapshot.shape != (output_channels,):
        raise CompileError("bias must have one FP32 value per output channel")

    reduction_axes = tuple(axis for axis in range(rank) if axis != output_axis)
    all_zero = np.max(np.abs(weight_snapshot), axis=reduction_axes) == np.float32(0.0)
    constant_channels = [
        int(channel)
        for channel in np.flatnonzero(all_zero & (bias_snapshot != np.float32(0.0)))
    ]
    try:
        selected_scale = normalize_scale_float32(
            output_qparams.scale / input_qparams.scale
        )
    except ValueError as error:
        if constant_channels:
            raise CompileError(
                "constant-channel weight scale cannot be represented as float32"
            ) from error
        selected_scale = 1.0
    channel_scales = {channel: selected_scale for channel in constant_channels}
    quantized_weight = _quantize_weights_per_channel(
        weight_snapshot,
        bias_snapshot,
        layout=layout,
        constant_channel_scales=channel_scales,
    )

    exact_codes: dict[int, int] = {}
    expected_output_codes: dict[int, int] = {}
    for channel in constant_channels:
        centered = round_half_away_from_zero(
            float(bias_snapshot[channel]) / output_qparams.scale
        )
        output_code = min(127, max(-128, centered + output_qparams.zero_point))
        expected_output_codes[channel] = output_code
        exact_codes[channel] = output_code - output_qparams.zero_point
    quantized_bias = _quantize_bias_int32(
        bias_snapshot,
        input_scale=input_qparams.scale,
        weight_qparams=quantized_weight.qparams,
        exact_channel_codes=exact_codes,
    )

    for channel in constant_channels:
        real_multiplier = (
            input_qparams.scale
            * quantized_weight.qparams.scales[channel]
            / output_qparams.scale
        )
        multiplier, shift = quantize_multiplier(real_multiplier)
        bias_code = int(quantized_bias.values[channel])
        if shift > 0 and abs(bias_code) * (1 << shift) > INT32_MAX:
            raise CompileError(
                f"constant channel {channel}: requantization left shift is not int32-safe"
            )
        try:
            requantized = multiply_by_quantized_multiplier(bias_code, multiplier, shift)
        except OverflowError as error:
            raise CompileError(
                f"constant channel {channel}: requantization is not int32-safe"
            ) from error
        actual = min(
            127,
            max(-128, requantized + output_qparams.zero_point),
        )
        if actual != expected_output_codes[channel]:
            raise CompileError(
                f"constant channel {channel}: selected qparams do not reproduce the "
                "directly quantized FP32 bias"
            )
    return quantized_weight, quantized_bias


__all__ = [
    "QuantizedBias",
    "QuantizedWeights",
    "quantize_bias_int32",
    "quantize_compute_constants",
    "quantize_weights_per_channel",
]
