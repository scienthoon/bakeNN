from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Protocol, runtime_checkable


@runtime_checkable
class Op(Protocol):
    """Structural contract shared by every immutable quantized IR operation."""

    name: str

    @property
    def inputs(self) -> tuple[str, ...]: ...

    @property
    def outputs(self) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class LinearOp:
    """INT8 fully-connected operation.

    The legacy named fields and positional constructor remain public.  Generic
    compiler passes consume ``inputs`` and ``outputs`` instead.
    """

    name: str
    input: str
    weight: str
    bias: str
    output: str
    activation_min: int = -128
    activation_max: int = 127

    def __post_init__(self) -> None:
        for field_name in ("activation_min", "activation_max"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{field_name} must be an integer")
            object.__setattr__(self, field_name, int(value))

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input, self.weight, self.bias)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)
