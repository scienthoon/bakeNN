from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from bakenn.ir.types import TARGET_DIM_MAX


@dataclass(frozen=True)
class ReshapeOp:
    name: str
    input: str
    output: str

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


@dataclass(frozen=True)
class FlattenOp:
    name: str
    input: str
    output: str

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


@dataclass(frozen=True)
class SliceOp:
    """One statically normalized positive-step slice along one axis."""

    name: str
    input: str
    output: str
    axis: int
    start: int
    stop: int
    step: int = 1

    def __post_init__(self) -> None:
        for name in ("axis", "start", "stop", "step"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"Slice {name} must be an integer")
            object.__setattr__(self, name, int(value))
        if self.axis < 0 or self.start < 0 or self.stop <= self.start or self.step <= 0:
            raise ValueError("Slice requires normalized axis/start/stop and a positive step")
        if self.step > TARGET_DIM_MAX:
            raise ValueError("Slice step exceeds the portable target dimension limit")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


@dataclass(frozen=True)
class ConcatenateOp:
    name: str
    input_names: tuple[str, ...]
    output: str
    axis: int

    def __post_init__(self) -> None:
        input_names = tuple(self.input_names)
        if len(input_names) < 2 or not all(isinstance(item, str) and item for item in input_names):
            raise ValueError("Concatenate requires at least two named inputs")
        if isinstance(self.axis, bool) or not isinstance(self.axis, Integral):
            raise ValueError("Concatenate axis must be an integer")
        object.__setattr__(self, "input_names", input_names)
        object.__setattr__(self, "axis", int(self.axis))

    @property
    def inputs(self) -> tuple[str, ...]:
        return self.input_names

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


__all__ = ["ConcatenateOp", "FlattenOp", "ReshapeOp", "SliceOp"]
