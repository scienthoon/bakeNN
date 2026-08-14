from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np

from bakenn.errors import CompileError
from bakenn.ir.types import PerTensorQParams, normalize_scale_float32
from bakenn.quantization.fixedpoint import round_half_away_from_zero


def _float32_values(values: object, description: str) -> np.ndarray:
    """Take a private C-contiguous FP32 snapshot and validate it eagerly."""

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
        raise CompileError(f"{description} must contain at least one sample")
    if not np.all(np.isfinite(snapshot)):
        raise CompileError(f"{description} contains NaN or infinity in FP32 semantics")
    snapshot.setflags(write=False)
    return snapshot


@dataclass(frozen=True)
class MinMaxObserver:
    """Persistent deterministic FP32 min/max observer for one graph edge.

    ``observe`` returns a new value rather than mutating shared state.  This
    makes calibration order explicit and prevents a frontend from accidentally
    reusing statistics between graph edges.
    """

    minimum: float | None = None
    maximum: float | None = None
    observed_elements: int = 0
    observed_batches: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.observed_elements, bool) or not isinstance(self.observed_elements, Integral):
            raise ValueError("observed_elements must be an integer")
        if isinstance(self.observed_batches, bool) or not isinstance(self.observed_batches, Integral):
            raise ValueError("observed_batches must be an integer")
        elements = int(self.observed_elements)
        batches = int(self.observed_batches)
        if elements < 0 or batches < 0:
            raise ValueError("observer counts cannot be negative")
        empty = self.minimum is None and self.maximum is None
        if empty != (elements == 0 and batches == 0):
            raise ValueError("empty observer statistics and counts are inconsistent")
        if (self.minimum is None) != (self.maximum is None):
            raise ValueError("observer minimum and maximum must both be set or both be absent")
        if not empty:
            if (
                isinstance(self.minimum, bool)
                or not isinstance(self.minimum, Real)
                or isinstance(self.maximum, bool)
                or not isinstance(self.maximum, Real)
            ):
                raise ValueError("observer extrema must be real numbers")
            minimum = float(self.minimum)  # type: ignore[arg-type]
            maximum = float(self.maximum)  # type: ignore[arg-type]
            if not math.isfinite(minimum) or not math.isfinite(maximum):
                raise ValueError("observer extrema must be finite")
            if minimum > maximum:
                raise ValueError("observer minimum cannot exceed maximum")
            if elements == 0 or batches == 0:
                raise ValueError("non-empty observer statistics require positive counts")
            object.__setattr__(self, "minimum", minimum)
            object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "observed_elements", elements)
        object.__setattr__(self, "observed_batches", batches)

    @property
    def is_observed(self) -> bool:
        return self.observed_batches > 0

    def observe(self, values: object) -> MinMaxObserver:
        snapshot = _float32_values(values, "calibration edge")
        batch_minimum = float(np.min(snapshot))
        batch_maximum = float(np.max(snapshot))
        minimum = batch_minimum if self.minimum is None else min(self.minimum, batch_minimum)
        maximum = batch_maximum if self.maximum is None else max(self.maximum, batch_maximum)
        return MinMaxObserver(
            minimum=minimum,
            maximum=maximum,
            observed_elements=self.observed_elements + int(snapshot.size),
            observed_batches=self.observed_batches + 1,
        )

    def activation_qparams(self, *, relu: bool = False) -> PerTensorQParams:
        """Choose affine INT8 qparams, always representing real zero exactly."""

        if not isinstance(relu, bool):
            raise CompileError("relu observer policy must be boolean")
        if self.minimum is None or self.maximum is None:
            raise CompileError("activation qparams require at least one observed calibration sample")
        minimum = 0.0 if relu else min(self.minimum, 0.0)
        maximum = max(self.maximum, 0.0)
        if minimum == maximum:
            # The only possible collapsed range after including zero is [0, 0].
            return PerTensorQParams(scale=1.0, zero_point=0)
        raw_scale = (maximum - minimum) / 255.0
        try:
            scale = normalize_scale_float32(raw_scale)
        except ValueError as error:
            raise CompileError("observed activation range cannot be represented by INT8") from error
        zero_point = round_half_away_from_zero(-128.0 - minimum / scale)
        zero_point = min(127, max(-128, zero_point))
        return PerTensorQParams(scale=scale, zero_point=zero_point)


def observe_minmax(batches: object) -> MinMaxObserver:
    """Observe an iterable of edge batches using immutable FP32 statistics."""

    try:
        iterator = iter(batches)  # type: ignore[arg-type]
    except TypeError as error:
        raise CompileError("calibration batches must be iterable") from error
    observer = MinMaxObserver()
    for batch in iterator:
        observer = observer.observe(batch)
    if not observer.is_observed:
        raise CompileError("calibration data must contain at least one sample")
    return observer


__all__ = ["MinMaxObserver", "observe_minmax"]
