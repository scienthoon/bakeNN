from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class SoftmaxStep:
    kernel_kind: ClassVar[str] = "softmax_s8_q15"

    name: str
    input: str
    output: str
    lut: tuple[int, ...]
    row_count: int
    class_count: int
    sum_bound: int
    arithmetic_profile: str = "bakenn.softmax_lut.q15.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "lut", tuple(self.lut))

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


__all__ = ["SoftmaxStep"]
