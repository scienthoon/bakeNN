from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral


def _integer(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{description} must be an integer")
    return int(value)


def _integer_tuple(value: tuple[int, ...], length: int, description: str) -> tuple[int, ...]:
    normalized = tuple(value)
    if len(normalized) != length:
        raise ValueError(f"{description} must contain {length} integers")
    return tuple(_integer(item, description) for item in normalized)


@dataclass(frozen=True)
class Conv2DOp:
    """Static batch-one NHWC/OHWI INT8 convolution."""

    name: str
    input: str
    weight: str
    bias: str
    output: str
    stride: tuple[int, int] = (1, 1)
    dilation: tuple[int, int] = (1, 1)
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)
    groups: int = 1
    activation_min: int = -128
    activation_max: int = 127

    def __post_init__(self) -> None:
        object.__setattr__(self, "stride", _integer_tuple(self.stride, 2, "stride"))
        object.__setattr__(self, "dilation", _integer_tuple(self.dilation, 2, "dilation"))
        object.__setattr__(self, "padding", _integer_tuple(self.padding, 4, "padding"))
        object.__setattr__(self, "groups", _integer(self.groups, "groups"))
        object.__setattr__(
            self, "activation_min", _integer(self.activation_min, "activation_min")
        )
        object.__setattr__(
            self, "activation_max", _integer(self.activation_max, "activation_max")
        )

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input, self.weight, self.bias)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


@dataclass(frozen=True)
class DepthwiseConv2DOp:
    """Static batch-one NHWC/HWO INT8 depthwise convolution."""

    name: str
    input: str
    weight: str
    bias: str
    output: str
    depth_multiplier: int = 1
    stride: tuple[int, int] = (1, 1)
    dilation: tuple[int, int] = (1, 1)
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)
    activation_min: int = -128
    activation_max: int = 127

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "depth_multiplier", _integer(self.depth_multiplier, "depth_multiplier")
        )
        object.__setattr__(self, "stride", _integer_tuple(self.stride, 2, "stride"))
        object.__setattr__(self, "dilation", _integer_tuple(self.dilation, 2, "dilation"))
        object.__setattr__(self, "padding", _integer_tuple(self.padding, 4, "padding"))
        object.__setattr__(
            self, "activation_min", _integer(self.activation_min, "activation_min")
        )
        object.__setattr__(
            self, "activation_max", _integer(self.activation_max, "activation_max")
        )

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input, self.weight, self.bias)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


__all__ = ["Conv2DOp", "DepthwiseConv2DOp"]
