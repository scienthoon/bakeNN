from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from bakenn.plan.types import AliasSpec


@dataclass(frozen=True)
class Pad2DStep:
    kernel_kind: ClassVar[str] = "pad2d_s8"
    name: str
    input: str
    output: str
    padding: tuple[int, int, int, int]
    arithmetic_profile: str = "bakenn.int8.pad2d.v1"

    @property
    def inputs(self) -> tuple[str, ...]: return (self.input,)
    @property
    def outputs(self) -> tuple[str, ...]: return (self.output,)
    @property
    def constants(self) -> tuple[str, ...]: return ()
    @property
    def aliases(self) -> tuple[AliasSpec, ...]: return ()
    @property
    def scratch_size(self) -> int: return 0
    @property
    def scratch_alignment(self) -> int: return 1


@dataclass(frozen=True)
class ReduceMeanStep:
    kernel_kind: ClassVar[str] = "reduce_mean_s8"
    name: str
    input: str
    output: str
    position_count: int
    channels: int
    multiplier: int
    shift: int
    accumulator_bound: int
    arithmetic_profile: str = "bakenn.int8.reduce_mean.v1"

    @property
    def inputs(self) -> tuple[str, ...]: return (self.input,)
    @property
    def outputs(self) -> tuple[str, ...]: return (self.output,)
    @property
    def constants(self) -> tuple[str, ...]: return ()
    @property
    def aliases(self) -> tuple[AliasSpec, ...]: return ()
    @property
    def scratch_size(self) -> int: return 0
    @property
    def scratch_alignment(self) -> int: return 1


__all__ = ["Pad2DStep", "ReduceMeanStep"]
