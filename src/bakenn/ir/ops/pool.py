from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral


def _pair(value: tuple[int, int], description: str) -> tuple[int, int]:
    normalized = tuple(value)
    if len(normalized) != 2 or any(
        isinstance(item, bool) or not isinstance(item, Integral) or item <= 0
        for item in normalized
    ):
        raise ValueError(f"{description} must contain two positive integers")
    return (int(normalized[0]), int(normalized[1]))


def _padding(value: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    normalized = tuple(value)
    if len(normalized) != 4 or any(
        isinstance(item, bool) or not isinstance(item, Integral) or item < 0
        for item in normalized
    ):
        raise ValueError("padding must contain four non-negative integers")
    return tuple(int(item) for item in normalized)  # type: ignore[return-value]


def _clamp(low: int, high: int) -> tuple[int, int]:
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in (low, high)):
        raise ValueError("activation clamp bounds must be integers")
    normalized = (int(low), int(high))
    if not -128 <= normalized[0] <= normalized[1] <= 127:
        raise ValueError("activation clamp must stay in int8 range")
    return normalized


@dataclass(frozen=True)
class AveragePool2DOp:
    name: str
    input: str
    output: str
    kernel: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)
    activation_min: int = -128
    activation_max: int = 127

    def __post_init__(self) -> None:
        object.__setattr__(self, "kernel", _pair(self.kernel, "kernel"))
        object.__setattr__(self, "stride", _pair(self.stride, "stride"))
        object.__setattr__(self, "padding", _padding(self.padding))
        low, high = _clamp(self.activation_min, self.activation_max)
        object.__setattr__(self, "activation_min", low)
        object.__setattr__(self, "activation_max", high)

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


@dataclass(frozen=True)
class MaxPool2DOp:
    name: str
    input: str
    output: str
    kernel: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)
    activation_min: int = -128
    activation_max: int = 127

    def __post_init__(self) -> None:
        object.__setattr__(self, "kernel", _pair(self.kernel, "kernel"))
        object.__setattr__(self, "stride", _pair(self.stride, "stride"))
        object.__setattr__(self, "padding", _padding(self.padding))
        low, high = _clamp(self.activation_min, self.activation_max)
        object.__setattr__(self, "activation_min", low)
        object.__setattr__(self, "activation_max", high)

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


__all__ = ["AveragePool2DOp", "MaxPool2DOp"]
