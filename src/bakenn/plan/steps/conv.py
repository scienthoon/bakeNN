from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from bakenn.plan.types import AliasSpec


@dataclass(frozen=True)
class Conv2DStep:
    kernel_kind: ClassVar[str] = "conv2d_s8"

    name: str
    input: str
    weight: str
    bias: str
    output: str
    stride: tuple[int, int]
    dilation: tuple[int, int]
    padding: tuple[int, int, int, int]
    groups: int
    multipliers: tuple[int, ...]
    shifts: tuple[int, ...]
    activation_min: int
    activation_max: int
    accumulator_bounds: tuple[int, ...]
    arithmetic_profile: str = "bakenn.int8.conv2d.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "stride", tuple(self.stride))
        object.__setattr__(self, "dilation", tuple(self.dilation))
        object.__setattr__(self, "padding", tuple(self.padding))
        object.__setattr__(self, "multipliers", tuple(self.multipliers))
        object.__setattr__(self, "shifts", tuple(self.shifts))
        object.__setattr__(self, "accumulator_bounds", tuple(self.accumulator_bounds))

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)

    @property
    def constants(self) -> tuple[str, ...]:
        return (self.weight, self.bias)

    @property
    def aliases(self) -> tuple[AliasSpec, ...]:
        return ()

    @property
    def scratch_size(self) -> int:
        return 0

    @property
    def scratch_alignment(self) -> int:
        return 1


@dataclass(frozen=True)
class DepthwiseConv2DStep:
    kernel_kind: ClassVar[str] = "depthwise_conv2d_s8"

    name: str
    input: str
    weight: str
    bias: str
    output: str
    depth_multiplier: int
    stride: tuple[int, int]
    dilation: tuple[int, int]
    padding: tuple[int, int, int, int]
    multipliers: tuple[int, ...]
    shifts: tuple[int, ...]
    activation_min: int
    activation_max: int
    accumulator_bounds: tuple[int, ...]
    arithmetic_profile: str = "bakenn.int8.depthwise_conv2d.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "stride", tuple(self.stride))
        object.__setattr__(self, "dilation", tuple(self.dilation))
        object.__setattr__(self, "padding", tuple(self.padding))
        object.__setattr__(self, "multipliers", tuple(self.multipliers))
        object.__setattr__(self, "shifts", tuple(self.shifts))
        object.__setattr__(self, "accumulator_bounds", tuple(self.accumulator_bounds))

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)

    @property
    def constants(self) -> tuple[str, ...]:
        return (self.weight, self.bias)

    @property
    def aliases(self) -> tuple[AliasSpec, ...]:
        return ()

    @property
    def scratch_size(self) -> int:
        return 0

    @property
    def scratch_alignment(self) -> int:
        return 1


__all__ = ["Conv2DStep", "DepthwiseConv2DStep"]
