from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Integral, Real
import struct
from typing import TypeAlias


# P0 emits portable C for 32-bit MCUs.  Keeping every individual tensor and
# public size value in this range makes flattened indexing and generated `u`
# literals valid even when the host compiler itself uses a wider size_t.
TARGET_DIM_MAX = (1 << 31) - 1
TARGET_SIZE_MAX = (1 << 32) - 1


def normalize_scale_float32(value: Real) -> float:
    """Return the exact binary64 representation of a deployable float32 scale."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("quantization scale must be a real number")
    source = float(value)
    if not math.isfinite(source) or source <= 0.0:
        raise ValueError("quantization scale must be finite and positive")
    try:
        normalized = struct.unpack("!f", struct.pack("!f", source))[0]
    except OverflowError as error:
        raise ValueError("quantization scale must be representable as float32") from error
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError("quantization scale must be a positive finite float32 value")
    return normalized


class DType(str, Enum):
    INT8 = "int8"
    INT32 = "int32"

    @property
    def itemsize(self) -> int:
        return 1 if self is DType.INT8 else 4


class Layout(str, Enum):
    NHWC = "NHWC"
    NLC = "NLC"
    OHWI = "OHWI"
    OWI = "OWI"
    HWO = "HWO"
    NC = "NC"
    OI = "OI"
    C = "C"


@dataclass(frozen=True)
class PerTensorQParams:
    scale: float
    zero_point: int

    def __post_init__(self) -> None:
        normalized_scale = normalize_scale_float32(self.scale)
        if isinstance(self.zero_point, bool) or not isinstance(self.zero_point, Integral):
            raise ValueError("zero_point must be an integer")
        if not -128 <= self.zero_point <= 127:
            raise ValueError("int8 zero_point must be in [-128, 127]")
        object.__setattr__(self, "scale", normalized_scale)
        object.__setattr__(self, "zero_point", int(self.zero_point))


@dataclass(frozen=True)
class PerAxisQParams:
    scales: tuple[float, ...]
    zero_points: tuple[int, ...]
    axis: int

    def __post_init__(self) -> None:
        scales = tuple(self.scales)
        zero_points = tuple(self.zero_points)
        if not scales or len(scales) != len(zero_points):
            raise ValueError("per-axis scales and zero_points must be non-empty and equal length")
        try:
            normalized_scales = tuple(normalize_scale_float32(scale) for scale in scales)
        except ValueError as error:
            raise ValueError(f"invalid per-axis scale: {error}") from error
        if any(isinstance(zero_point, bool) or not isinstance(zero_point, Integral) for zero_point in zero_points):
            raise ValueError("per-axis zero_points must be integers")
        if any(not -128 <= zero_point <= 127 for zero_point in zero_points):
            raise ValueError("int8 per-axis zero_points must be in [-128, 127]")
        if isinstance(self.axis, bool) or not isinstance(self.axis, Integral) or self.axis < 0:
            raise ValueError("per-axis quantized dimension must be non-negative")
        object.__setattr__(self, "scales", normalized_scales)
        object.__setattr__(self, "zero_points", tuple(int(zero_point) for zero_point in zero_points))
        object.__setattr__(self, "axis", int(self.axis))


QParams: TypeAlias = PerTensorQParams | PerAxisQParams


@dataclass(frozen=True)
class TensorType:
    shape: tuple[int, ...]
    dtype: DType
    layout: Layout
    qparams: QParams

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        if not shape or any(
            isinstance(dim, bool) or not isinstance(dim, Integral) or dim <= 0 for dim in shape
        ):
            raise ValueError("tensor shapes must be static and strictly positive")
        if any(dim > TARGET_DIM_MAX for dim in shape):
            raise ValueError(f"tensor dimensions must not exceed {TARGET_DIM_MAX}")
        if not isinstance(self.dtype, DType) or not isinstance(self.layout, Layout):
            raise ValueError("tensor dtype and layout must use BakeNN enum values")
        if not isinstance(self.qparams, (PerTensorQParams, PerAxisQParams)):
            raise ValueError("tensor qparams must use a BakeNN quantization type")
        normalized_shape = tuple(int(dim) for dim in shape)
        if isinstance(self.qparams, PerAxisQParams):
            if self.qparams.axis >= len(normalized_shape):
                raise ValueError("per-axis qparams axis is outside the tensor rank")
            if len(self.qparams.scales) != normalized_shape[self.qparams.axis]:
                raise ValueError(
                    "per-axis qparam count must equal the quantized tensor dimension"
                )
        numel = math.prod(normalized_shape)
        if numel * self.dtype.itemsize > TARGET_SIZE_MAX:
            raise ValueError(
                f"tensor storage must not exceed the 32-bit target limit of {TARGET_SIZE_MAX} bytes"
            )
        object.__setattr__(self, "shape", normalized_shape)

    @property
    def numel(self) -> int:
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    @property
    def nbytes(self) -> int:
        return self.numel * self.dtype.itemsize


__all__ = [
    "DType",
    "Layout",
    "PerAxisQParams",
    "PerTensorQParams",
    "QParams",
    "TARGET_DIM_MAX",
    "TARGET_SIZE_MAX",
    "TensorType",
    "normalize_scale_float32",
]
