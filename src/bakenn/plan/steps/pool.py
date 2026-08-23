from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from bakenn.ir.ops.pool import (
    AVERAGE_POOL_PROFILE_BAKENN_V1,
    AVERAGE_POOL_PROFILES,
)


@dataclass(frozen=True)
class AveragePool2DStep:
    kernel_kind: ClassVar[str] = "average_pool2d_s8"

    name: str
    input: str
    output: str
    kernel: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int, int, int]
    activation_min: int
    activation_max: int
    accumulator_bound: int
    arithmetic_profile: str = AVERAGE_POOL_PROFILE_BAKENN_V1

    def __post_init__(self) -> None:
        if self.arithmetic_profile not in AVERAGE_POOL_PROFILES:
            raise ValueError(
                f"unsupported AveragePool2D arithmetic profile: {self.arithmetic_profile}"
            )

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
    def aliases(self) -> tuple[object, ...]:
        return ()

    @property
    def scratch_size(self) -> int:
        return 0

    @property
    def scratch_alignment(self) -> int:
        return 1


@dataclass(frozen=True)
class MaxPool2DStep:
    kernel_kind: ClassVar[str] = "max_pool2d_s8"

    name: str
    input: str
    output: str
    kernel: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int, int, int]
    activation_min: int
    activation_max: int
    arithmetic_profile: str = "bakenn.int8.max_pool2d.v1"

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
    def aliases(self) -> tuple[object, ...]:
        return ()

    @property
    def scratch_size(self) -> int:
        return 0

    @property
    def scratch_alignment(self) -> int:
        return 1


__all__ = ["AveragePool2DStep", "MaxPool2DStep"]
