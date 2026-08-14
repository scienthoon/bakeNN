from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _clamp(low: object, high: object) -> tuple[int, int]:
    low_i = _integer(low, "activation_min", minimum=-128)
    high_i = _integer(high, "activation_max", minimum=-128)
    if not -128 <= low_i <= high_i <= 127:
        raise ValueError("activation clamp must stay within int8")
    return low_i, high_i


@dataclass(frozen=True)
class Conv1DOp:
    name: str
    input: str
    weight: str
    bias: str
    output: str
    stride: int = 1
    dilation: int = 1
    padding: tuple[int, int] = (0, 0)
    groups: int = 1
    activation_min: int = -128
    activation_max: int = 127

    def __post_init__(self) -> None:
        padding = tuple(self.padding)
        if len(padding) != 2:
            raise ValueError("Conv1D padding must be (left,right)")
        object.__setattr__(self, "stride", _integer(self.stride, "stride", minimum=1))
        object.__setattr__(self, "dilation", _integer(self.dilation, "dilation", minimum=1))
        object.__setattr__(self, "groups", _integer(self.groups, "groups", minimum=1))
        object.__setattr__(self, "padding", tuple(_integer(x, "padding") for x in padding))
        low, high = _clamp(self.activation_min, self.activation_max)
        object.__setattr__(self, "activation_min", low)
        object.__setattr__(self, "activation_max", high)

    @property
    def inputs(self) -> tuple[str, ...]: return (self.input, self.weight, self.bias)
    @property
    def outputs(self) -> tuple[str, ...]: return (self.output,)


@dataclass(frozen=True)
class AveragePool1DOp:
    name: str
    input: str
    output: str
    kernel: int
    stride: int
    padding: tuple[int, int] = (0, 0)
    activation_min: int = -128
    activation_max: int = 127

    def __post_init__(self) -> None:
        padding = tuple(self.padding)
        if len(padding) != 2:
            raise ValueError("Pool1D padding must be (left,right)")
        object.__setattr__(self, "kernel", _integer(self.kernel, "kernel", minimum=1))
        object.__setattr__(self, "stride", _integer(self.stride, "stride", minimum=1))
        object.__setattr__(self, "padding", tuple(_integer(x, "padding") for x in padding))
        low, high = _clamp(self.activation_min, self.activation_max)
        object.__setattr__(self, "activation_min", low)
        object.__setattr__(self, "activation_max", high)

    @property
    def inputs(self) -> tuple[str, ...]: return (self.input,)
    @property
    def outputs(self) -> tuple[str, ...]: return (self.output,)


@dataclass(frozen=True)
class MaxPool1DOp(AveragePool1DOp):
    pass


__all__ = ["AveragePool1DOp", "Conv1DOp", "MaxPool1DOp"]
