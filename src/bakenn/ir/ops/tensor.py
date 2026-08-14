from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral


def _integers(values: tuple[int, ...], description: str) -> tuple[int, ...]:
    normalized = tuple(values)
    if not normalized or any(isinstance(value, bool) or not isinstance(value, Integral) for value in normalized):
        raise ValueError(f"{description} must contain integers")
    return tuple(int(value) for value in normalized)


@dataclass(frozen=True)
class Pad2DOp:
    name: str
    input: str
    output: str
    padding: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        padding = _integers(self.padding, "padding")
        if len(padding) != 4 or any(value < 0 for value in padding):
            raise ValueError("Pad2D padding must be (top,bottom,left,right) and non-negative")
        object.__setattr__(self, "padding", padding)

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


@dataclass(frozen=True)
class ReduceMeanOp:
    name: str
    input: str
    output: str
    axes: tuple[int, ...]
    keepdims: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", _integers(self.axes, "ReduceMean axes"))
        if not isinstance(self.keepdims, bool):
            raise ValueError("ReduceMean keepdims must be boolean")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


__all__ = ["Pad2DOp", "ReduceMeanOp"]
