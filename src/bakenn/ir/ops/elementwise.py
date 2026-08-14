from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral


def _normalize_clamp(instance: object) -> None:
    for field_name in ("activation_min", "activation_max"):
        value = getattr(instance, field_name)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{field_name} must be an integer")
        object.__setattr__(instance, field_name, int(value))
    if not -128 <= instance.activation_min <= instance.activation_max <= 127:  # type: ignore[attr-defined]
        raise ValueError("activation clamp must satisfy -128 <= min <= max <= 127")


def _normalize_inplace(instance: object) -> None:
    if not isinstance(getattr(instance, "inplace"), bool):
        raise ValueError("inplace must be a boolean")


@dataclass(frozen=True)
class AddOp:
    """Affine-INT8 addition with verified static same-rank broadcasting."""

    name: str
    input_a: str
    input_b: str
    output: str
    activation_min: int = -128
    activation_max: int = 127

    def __post_init__(self) -> None:
        _normalize_clamp(self)

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input_a, self.input_b)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


@dataclass(frozen=True)
class MulOp:
    """Affine-INT8 multiplication with verified static same-rank broadcasting."""

    name: str
    input_a: str
    input_b: str
    output: str
    activation_min: int = -128
    activation_max: int = 127

    def __post_init__(self) -> None:
        _normalize_clamp(self)

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input_a, self.input_b)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


@dataclass(frozen=True)
class ClampOp:
    """Raw-code clamp for equal-qparam INT8 tensors.

    ``inplace`` is deliberately opt-in.  The memory planner still proves the
    requested alias safe; merely constructing a Clamp does not imply an alias.
    """

    name: str
    input: str
    output: str
    activation_min: int
    activation_max: int
    inplace: bool = False

    def __post_init__(self) -> None:
        _normalize_clamp(self)
        _normalize_inplace(self)

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


@dataclass(frozen=True)
class RequantizeOp:
    """Internal per-tensor affine requantization operation."""

    name: str
    input: str
    output: str
    inplace: bool = False
    activation_min: int = -128
    activation_max: int = 127

    def __post_init__(self) -> None:
        _normalize_inplace(self)
        _normalize_clamp(self)

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


__all__ = ["AddOp", "ClampOp", "MulOp", "RequantizeOp"]
