from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from bakenn.plan.types import AliasSpec


@dataclass(frozen=True)
class LUTActivationStep:
    kernel_kind: ClassVar[str] = "activation_lut_s8"

    name: str
    input: str
    output: str
    operation: str
    lut: tuple[int, ...]
    arithmetic_profile: str = "bakenn.int8.lut.v1"

    def __post_init__(self) -> None:
        values = tuple(self.lut)
        if self.operation not in ("sigmoid", "hardsigmoid", "hardswish", "silu"):
            raise ValueError("unknown LUT activation")
        if len(values) != 256 or any(not -128 <= value <= 127 for value in values):
            raise ValueError("an int8 activation LUT must contain exactly 256 codes")
        object.__setattr__(self, "lut", values)

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)

    @property
    def constants(self) -> tuple[str, ...]:
        return ()

    @property
    def aliases(self) -> tuple[AliasSpec, ...]:
        return ()

    @property
    def scratch_size(self) -> int:
        return 0

    @property
    def scratch_alignment(self) -> int:
        return 1


__all__ = ["LUTActivationStep"]
