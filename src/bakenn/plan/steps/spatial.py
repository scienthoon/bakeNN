from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import ClassVar

from bakenn.plan.types import AliasSpec


class _NoStorage:
    @property
    def constants(self) -> tuple[str, ...]: return ()
    @property
    def aliases(self) -> tuple[AliasSpec, ...]: return ()
    @property
    def scratch_size(self) -> int: return 0
    @property
    def scratch_alignment(self) -> int: return 1


@dataclass(frozen=True)
class ResizeNearest2DStep(_NoStorage):
    kernel_kind: ClassVar[str] = "resize_nearest2d_s8"
    name: str
    input: str
    output: str
    y_indices: tuple[int, ...]
    x_indices: tuple[int, ...]
    arithmetic_profile: str = "bakenn.int8.resize_nearest2d.v1"

    def __post_init__(self) -> None:
        for name in ("y_indices", "x_indices"):
            values = tuple(int(value) for value in getattr(self, name))
            if not values or any(value < 0 for value in values):
                raise ValueError("nearest Resize2D maps must be non-empty and non-negative")
            object.__setattr__(self, name, values)
    @property
    def inputs(self) -> tuple[str, ...]: return (self.input,)
    @property
    def outputs(self) -> tuple[str, ...]: return (self.output,)


@dataclass(frozen=True)
class ResizeBilinear2DStep(_NoStorage):
    kernel_kind: ClassVar[str] = "resize_bilinear2d_s8"
    name: str
    input: str
    output: str
    align_corners: bool
    y0: tuple[int, ...]
    y1: tuple[int, ...]
    yw_q15: tuple[int, ...]
    x0: tuple[int, ...]
    x1: tuple[int, ...]
    xw_q15: tuple[int, ...]
    arithmetic_profile: str = "bakenn.int8.resize_bilinear.q15.v1"

    def __post_init__(self) -> None:
        for name in ("y0", "y1", "yw_q15", "x0", "x1", "xw_q15"):
            object.__setattr__(self, name, tuple(int(value) for value in getattr(self, name)))
        if not isinstance(self.align_corners, bool):
            raise ValueError("align_corners must be boolean")
        if len(self.y0) != len(self.y1) or len(self.y0) != len(self.yw_q15):
            raise ValueError("bilinear y maps must have equal length")
        if len(self.x0) != len(self.x1) or len(self.x0) != len(self.xw_q15):
            raise ValueError("bilinear x maps must have equal length")
        if any(not 0 <= value <= 32768 for value in (*self.yw_q15, *self.xw_q15)):
            raise ValueError("bilinear Q15 weights must be in [0, 32768]")

    @property
    def inputs(self) -> tuple[str, ...]: return (self.input,)
    @property
    def outputs(self) -> tuple[str, ...]: return (self.output,)


@dataclass(frozen=True)
class ConvTranspose2DStep:
    kernel_kind: ClassVar[str] = "conv_transpose2d_s8"
    name: str
    input: str
    weight: str
    bias: str
    output: str
    stride: tuple[int, int]
    dilation: tuple[int, int]
    padding: tuple[int, int, int, int]
    output_padding: tuple[int, int]
    groups: int
    multipliers: tuple[int, ...]
    shifts: tuple[int, ...]
    activation_min: int
    activation_max: int
    accumulator_bounds: tuple[int, ...]
    arithmetic_profile: str = "bakenn.int8.conv_transpose2d.v1"

    def __post_init__(self) -> None:
        for name in ("stride", "dilation", "padding", "output_padding", "multipliers", "shifts", "accumulator_bounds"):
            object.__setattr__(self, name, tuple(int(value) for value in getattr(self, name)))
        if isinstance(self.groups, bool) or not isinstance(self.groups, Integral) or self.groups <= 0:
            raise ValueError("ConvTranspose2D groups must be a positive integer")
        object.__setattr__(self, "groups", int(self.groups))

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


__all__ = ["ConvTranspose2DStep", "ResizeBilinear2DStep", "ResizeNearest2DStep"]
