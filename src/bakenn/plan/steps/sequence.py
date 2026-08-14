from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from bakenn.plan.types import AliasSpec


@dataclass(frozen=True)
class Conv1DStep:
    kernel_kind: ClassVar[str] = "conv1d_s8"
    name: str
    input: str
    weight: str
    bias: str
    output: str
    stride: int
    dilation: int
    padding: tuple[int, int]
    groups: int
    multipliers: tuple[int, ...]
    shifts: tuple[int, ...]
    activation_min: int
    activation_max: int
    accumulator_bounds: tuple[int, ...]
    arithmetic_profile: str = "bakenn.int8.conv1d.v1"

    @property
    def inputs(self) -> tuple[str, ...]: return (self.input,)
    @property
    def outputs(self) -> tuple[str, ...]: return (self.output,)
    @property
    def constants(self) -> tuple[str, ...]: return (self.weight, self.bias)
    @property
    def aliases(self) -> tuple[AliasSpec, ...]: return ()
    @property
    def scratch_size(self) -> int: return 0
    @property
    def scratch_alignment(self) -> int: return 1


@dataclass(frozen=True)
class AveragePool1DStep:
    kernel_kind: ClassVar[str] = "average_pool1d_s8"
    name: str
    input: str
    output: str
    kernel: int
    stride: int
    padding: tuple[int, int]
    activation_min: int
    activation_max: int
    accumulator_bound: int
    arithmetic_profile: str = "bakenn.int8.average_pool1d.v1"

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
class MaxPool1DStep(AveragePool1DStep):
    kernel_kind: ClassVar[str] = "max_pool1d_s8"
    arithmetic_profile: str = "bakenn.int8.max_pool1d.v1"


__all__ = ["AveragePool1DStep", "Conv1DStep", "MaxPool1DStep"]
