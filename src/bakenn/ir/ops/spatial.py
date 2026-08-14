from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral


def _pair(value: object, name: str, *, minimum: int = 0) -> tuple[int, int]:
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{name} must contain two integers") from error
    if len(items) != 2 or any(
        isinstance(item, bool) or not isinstance(item, Integral) or int(item) < minimum
        for item in items
    ):
        raise ValueError(f"{name} must contain two integers >= {minimum}")
    return int(items[0]), int(items[1])


def _padding(value: object) -> tuple[int, int, int, int]:
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("padding must contain four integers") from error
    if len(items) != 4 or any(
        isinstance(item, bool) or not isinstance(item, Integral) or int(item) < 0
        for item in items
    ):
        raise ValueError("padding must contain four non-negative integers")
    return tuple(int(item) for item in items)  # type: ignore[return-value]


@dataclass(frozen=True)
class ResizeNearest2DOp:
    name: str
    input: str
    output: str

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


@dataclass(frozen=True)
class ResizeBilinear2DOp:
    name: str
    input: str
    output: str
    align_corners: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.align_corners, bool):
            raise ValueError("align_corners must be boolean")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


@dataclass(frozen=True)
class ConvTranspose2DOp:
    """Static batch-one grouped NHWC/OHWI INT8 transposed convolution."""

    name: str
    input: str
    weight: str
    bias: str
    output: str
    stride: tuple[int, int] = (1, 1)
    dilation: tuple[int, int] = (1, 1)
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)
    output_padding: tuple[int, int] = (0, 0)
    groups: int = 1
    activation_min: int = -128
    activation_max: int = 127

    def __post_init__(self) -> None:
        object.__setattr__(self, "stride", _pair(self.stride, "stride", minimum=1))
        object.__setattr__(self, "dilation", _pair(self.dilation, "dilation", minimum=1))
        object.__setattr__(self, "padding", _padding(self.padding))
        object.__setattr__(
            self, "output_padding", _pair(self.output_padding, "output_padding")
        )
        if isinstance(self.groups, bool) or not isinstance(self.groups, Integral) or self.groups <= 0:
            raise ValueError("groups must be a positive integer")
        object.__setattr__(self, "groups", int(self.groups))
        for name in ("activation_min", "activation_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        if not -128 <= self.activation_min <= self.activation_max <= 127:
            raise ValueError("activation clamp must satisfy -128 <= min <= max <= 127")

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input, self.weight, self.bias)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


__all__ = ["ConvTranspose2DOp", "ResizeBilinear2DOp", "ResizeNearest2DOp"]
