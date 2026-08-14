from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from bakenn.plan.types import AliasKind, AliasSpec


@dataclass(frozen=True)
class ReshapeStep:
    kernel_kind: ClassVar[str] = "reshape_view"

    name: str
    input: str
    output: str
    materialize: bool = False
    arithmetic_profile: str = "bakenn.int8.reshape.v1"

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
        return () if self.materialize else (AliasSpec(self.output, self.input, AliasKind.VIEW),)

    @property
    def scratch_size(self) -> int:
        return 0

    @property
    def scratch_alignment(self) -> int:
        return 1


@dataclass(frozen=True)
class FlattenStep:
    kernel_kind: ClassVar[str] = "flatten_view"

    name: str
    input: str
    output: str
    materialize: bool = False
    arithmetic_profile: str = "bakenn.int8.flatten.v1"

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
        return () if self.materialize else (AliasSpec(self.output, self.input, AliasKind.VIEW),)

    @property
    def scratch_size(self) -> int:
        return 0

    @property
    def scratch_alignment(self) -> int:
        return 1


@dataclass(frozen=True)
class SliceStep:
    kernel_kind: ClassVar[str] = "slice_s8"

    name: str
    input: str
    output: str
    axis: int
    start: int
    step: int
    outer_size: int
    input_axis_size: int
    output_axis_size: int
    inner_size: int
    arithmetic_profile: str = "bakenn.int8.slice.v1"

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
class ConcatenateStep:
    kernel_kind: ClassVar[str] = "concatenate_s8"

    name: str
    input_names: tuple[str, ...]
    output: str
    axis: int
    outer_size: int
    inner_size: int
    axis_sizes: tuple[int, ...]
    arithmetic_profile: str = "bakenn.int8.concatenate.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_names", tuple(self.input_names))
        object.__setattr__(self, "axis_sizes", tuple(self.axis_sizes))

    @property
    def inputs(self) -> tuple[str, ...]:
        return self.input_names

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


__all__ = ["ConcatenateStep", "FlattenStep", "ReshapeStep", "SliceStep"]
