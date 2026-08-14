from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from bakenn.errors import CompileError


class FloatDType(str, Enum):
    FLOAT32 = "float32"


class FloatLayout(str, Enum):
    NCHW = "NCHW"
    NCL = "NCL"
    NC = "NC"
    OIHW = "OIHW"
    IOHW = "IOHW"
    OIW = "OIW"
    OI = "OI"
    C = "C"
    SCALAR = "SCALAR"


class FloatValueKind(str, Enum):
    INPUT = "input"
    PARAMETER = "parameter"
    BUFFER = "buffer"
    CONSTANT = "constant"
    ACTIVATION = "activation"


def _shape(value: object, description: str) -> tuple[int, ...]:
    try:
        dimensions = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{description} must be a static shape") from error
    if not dimensions or any(
        isinstance(dimension, bool)
        or not isinstance(dimension, Integral)
        or int(dimension) <= 0
        for dimension in dimensions
    ):
        raise ValueError(f"{description} must contain positive static dimensions")
    return tuple(int(dimension) for dimension in dimensions)


def _pair(value: object, description: str, *, positive: bool) -> tuple[int, int]:
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{description} must contain two integers") from error
    minimum = 1 if positive else 0
    if len(items) != 2 or any(
        isinstance(item, bool)
        or not isinstance(item, Integral)
        or int(item) < minimum
        for item in items
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{description} must contain two {qualifier} integers")
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
class FloatValue:
    name: str
    shape: tuple[int, ...]
    dtype: FloatDType
    layout: FloatLayout
    kind: FloatValueKind
    source_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("float value name must be non-empty")
        object.__setattr__(self, "shape", _shape(self.shape, f"{self.name} shape"))
        if not isinstance(self.dtype, FloatDType):
            raise ValueError("float value dtype must use FloatDType")
        if not isinstance(self.layout, FloatLayout):
            raise ValueError("float value layout must use FloatLayout")
        if not isinstance(self.kind, FloatValueKind):
            raise ValueError("float value kind must use FloatValueKind")
        if self.source_name is not None and (
            not isinstance(self.source_name, str) or not self.source_name
        ):
            raise ValueError("source_name must be a non-empty string when provided")

    @property
    def numel(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result


@runtime_checkable
class FloatOp(Protocol):
    name: str

    @property
    def inputs(self) -> tuple[str, ...]: ...

    @property
    def outputs(self) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class FloatConv2DOp:
    name: str
    input: str
    weight: str
    bias: str | None
    output: str
    stride: tuple[int, int]
    padding: tuple[int, int, int, int]
    dilation: tuple[int, int]
    groups: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "stride", _pair(self.stride, "stride", positive=True))
        object.__setattr__(self, "padding", _padding(self.padding))
        object.__setattr__(self, "dilation", _pair(self.dilation, "dilation", positive=True))
        if isinstance(self.groups, bool) or not isinstance(self.groups, Integral) or self.groups <= 0:
            raise ValueError("FloatConv2DOp groups must be positive")
        object.__setattr__(self, "groups", int(self.groups))

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input, self.weight) if self.bias is None else (self.input, self.weight, self.bias)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


@dataclass(frozen=True)
class FloatDepthwiseConv2DOp:
    name: str
    input: str
    weight: str
    bias: str | None
    output: str
    stride: tuple[int, int]
    padding: tuple[int, int, int, int]
    dilation: tuple[int, int]
    input_channels: int
    depth_multiplier: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "stride", _pair(self.stride, "stride", positive=True))
        object.__setattr__(self, "padding", _padding(self.padding))
        object.__setattr__(self, "dilation", _pair(self.dilation, "dilation", positive=True))
        for field_name in ("input_channels", "depth_multiplier"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
            object.__setattr__(self, field_name, int(value))

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input, self.weight) if self.bias is None else (self.input, self.weight, self.bias)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


@dataclass(frozen=True)
class FloatConv1DOp:
    name: str
    input: str
    weight: str
    bias: str | None
    output: str
    stride: int
    padding: tuple[int, int]
    dilation: int
    groups: int = 1

    def __post_init__(self) -> None:
        for field_name in ("stride", "dilation", "groups"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
            object.__setattr__(self, field_name, int(value))
        padding = tuple(self.padding)
        if len(padding) != 2 or any(
            isinstance(value, bool) or not isinstance(value, Integral) or value < 0
            for value in padding
        ):
            raise ValueError("Conv1D padding must contain two non-negative integers")
        object.__setattr__(self, "padding", tuple(int(value) for value in padding))

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input, self.weight) if self.bias is None else (self.input, self.weight, self.bias)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


@dataclass(frozen=True)
class FloatConvTranspose2DOp:
    name: str
    input: str
    weight: str
    bias: str | None
    output: str
    stride: tuple[int, int]
    padding: tuple[int, int, int, int]
    output_padding: tuple[int, int]
    dilation: tuple[int, int]
    groups: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "stride", _pair(self.stride, "stride", positive=True))
        object.__setattr__(self, "padding", _padding(self.padding))
        object.__setattr__(self, "output_padding", _pair(self.output_padding, "output_padding", positive=False))
        object.__setattr__(self, "dilation", _pair(self.dilation, "dilation", positive=True))
        if isinstance(self.groups, bool) or not isinstance(self.groups, Integral) or int(self.groups) <= 0:
            raise ValueError("FloatConvTranspose2DOp groups must be a positive integer")
        object.__setattr__(self, "groups", int(self.groups))

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input, self.weight) if self.bias is None else (self.input, self.weight, self.bias)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


@dataclass(frozen=True)
class FloatLinearOp:
    name: str
    input: str
    weight: str
    bias: str | None
    output: str

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input, self.weight) if self.bias is None else (self.input, self.weight, self.bias)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


@dataclass(frozen=True)
class FloatAddOp:
    name: str
    input_a: str
    input_b: str
    output: str

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input_a, self.input_b)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


@dataclass(frozen=True)
class FloatMulOp:
    name: str
    input_a: str
    input_b: str
    output: str

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input_a, self.input_b)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


@dataclass(frozen=True)
class FloatReLUOp:
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
class FloatReLU6Op(FloatReLUOp):
    pass


@dataclass(frozen=True)
class FloatSigmoidOp(FloatReLUOp):
    pass


@dataclass(frozen=True)
class FloatHardSwishOp(FloatReLUOp):
    pass


@dataclass(frozen=True)
class FloatHardSigmoidOp(FloatReLUOp):
    pass


@dataclass(frozen=True)
class FloatSiLUOp(FloatReLUOp):
    pass


@dataclass(frozen=True)
class _FloatPool2DOp:
    name: str
    input: str
    output: str
    kernel: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "kernel", _pair(self.kernel, "kernel", positive=True))
        object.__setattr__(self, "stride", _pair(self.stride, "stride", positive=True))
        object.__setattr__(self, "padding", _padding(self.padding))

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


@dataclass(frozen=True)
class FloatAveragePool2DOp(_FloatPool2DOp):
    pass


@dataclass(frozen=True)
class FloatMaxPool2DOp(_FloatPool2DOp):
    pass


@dataclass(frozen=True)
class _FloatPool1DOp:
    name: str
    input: str
    output: str
    kernel: int
    stride: int
    padding: tuple[int, int]

    def __post_init__(self) -> None:
        for field_name in ("kernel", "stride"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{field_name} must be positive")
            object.__setattr__(self, field_name, int(value))
        padding = tuple(self.padding)
        if len(padding) != 2 or any(
            isinstance(value, bool) or not isinstance(value, Integral) or value < 0
            for value in padding
        ):
            raise ValueError("Pool1D padding must contain two non-negative integers")
        object.__setattr__(self, "padding", tuple(int(value) for value in padding))

    @property
    def inputs(self) -> tuple[str, ...]: return (self.input,)
    @property
    def outputs(self) -> tuple[str, ...]: return (self.output,)


@dataclass(frozen=True)
class FloatAveragePool1DOp(_FloatPool1DOp):
    pass


@dataclass(frozen=True)
class FloatMaxPool1DOp(_FloatPool1DOp):
    pass


@dataclass(frozen=True)
class FloatPad2DOp:
    name: str
    input: str
    output: str
    padding: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "padding", _padding(self.padding))

    @property
    def inputs(self) -> tuple[str, ...]: return (self.input,)
    @property
    def outputs(self) -> tuple[str, ...]: return (self.output,)


@dataclass(frozen=True)
class FloatReduceMeanOp:
    name: str
    input: str
    output: str
    axes: tuple[int, ...]
    keepdims: bool

    def __post_init__(self) -> None:
        axes = tuple(self.axes)
        if not axes or any(isinstance(axis, bool) or not isinstance(axis, Integral) for axis in axes):
            raise ValueError("ReduceMean axes must be static integers")
        if not isinstance(self.keepdims, bool):
            raise ValueError("ReduceMean keepdims must be boolean")
        object.__setattr__(self, "axes", tuple(int(axis) for axis in axes))

    @property
    def inputs(self) -> tuple[str, ...]: return (self.input,)
    @property
    def outputs(self) -> tuple[str, ...]: return (self.output,)


@dataclass(frozen=True)
class FloatResizeNearest2DOp:
    name: str
    input: str
    output: str

    @property
    def inputs(self) -> tuple[str, ...]: return (self.input,)
    @property
    def outputs(self) -> tuple[str, ...]: return (self.output,)


@dataclass(frozen=True)
class FloatResizeBilinear2DOp:
    name: str
    input: str
    output: str
    align_corners: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.align_corners, bool):
            raise ValueError("align_corners must be boolean")

    @property
    def inputs(self) -> tuple[str, ...]: return (self.input,)
    @property
    def outputs(self) -> tuple[str, ...]: return (self.output,)


@dataclass(frozen=True)
class FloatFlattenOp:
    name: str
    input: str
    output: str
    start_dim: int = 1
    end_dim: int = -1

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


@dataclass(frozen=True)
class FloatReshapeOp:
    name: str
    input: str
    output: str
    target_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_shape", _shape(self.target_shape, "reshape target"))

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


@dataclass(frozen=True)
class FloatSliceOp:
    name: str
    input: str
    output: str
    axis: int
    start: int
    stop: int
    step: int = 1

    def __post_init__(self) -> None:
        for name in ("axis", "start", "stop", "step"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"FloatSlice {name} must be an integer")
            object.__setattr__(self, name, int(value))
        if self.axis < 0 or self.start < 0 or self.stop <= self.start or self.step <= 0:
            raise ValueError("FloatSlice requires normalized non-empty positive-step bounds")

    @property
    def inputs(self) -> tuple[str, ...]: return (self.input,)
    @property
    def outputs(self) -> tuple[str, ...]: return (self.output,)


@dataclass(frozen=True)
class FloatConcatOp:
    name: str
    input_values: tuple[str, ...]
    output: str
    axis: int

    def __post_init__(self) -> None:
        inputs = tuple(self.input_values)
        if len(inputs) < 2 or not all(isinstance(value, str) and value for value in inputs):
            raise ValueError("Concat requires at least two named inputs")
        if isinstance(self.axis, bool) or not isinstance(self.axis, Integral):
            raise ValueError("Concat axis must be an integer")
        object.__setattr__(self, "input_values", inputs)
        object.__setattr__(self, "axis", int(self.axis))

    @property
    def inputs(self) -> tuple[str, ...]:
        return self.input_values

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


@dataclass(frozen=True)
class FloatSoftmaxOp:
    name: str
    input: str
    output: str
    axis: int = -1

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input,)

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)


FloatOpTypes = (
    FloatConv1DOp,
    FloatConv2DOp,
    FloatConvTranspose2DOp,
    FloatDepthwiseConv2DOp,
    FloatLinearOp,
    FloatAddOp,
    FloatMulOp,
    FloatReLU6Op,
    FloatReLUOp,
    FloatSigmoidOp,
    FloatHardSigmoidOp,
    FloatHardSwishOp,
    FloatSiLUOp,
    FloatAveragePool1DOp,
    FloatMaxPool1DOp,
    FloatAveragePool2DOp,
    FloatMaxPool2DOp,
    FloatFlattenOp,
    FloatReshapeOp,
    FloatSliceOp,
    FloatConcatOp,
    FloatSoftmaxOp,
    FloatPad2DOp,
    FloatReduceMeanOp,
    FloatResizeNearest2DOp,
    FloatResizeBilinear2DOp,
)


@dataclass(frozen=True)
class FloatGraph:
    name: str
    values: Mapping[str, FloatValue]
    constants: Mapping[str, np.ndarray]
    ops: tuple[FloatOp, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    source: str = "torch.export"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("float graph name must be non-empty")
        values = MappingProxyType(dict(self.values))
        constants: dict[str, np.ndarray] = {}
        for name, array in self.constants.items():
            if name not in values:
                raise ValueError(f"constant {name} is missing a FloatValue")
            frozen = np.asarray(array, dtype=np.float32, order="C").copy(order="C")
            if frozen.shape != values[name].shape:
                raise ValueError(f"constant {name} shape does not match its FloatValue")
            if not np.all(np.isfinite(frozen)):
                raise CompileError(f"constant {name} contains NaN or infinity")
            frozen.setflags(write=False)
            constants[name] = frozen
        ops = tuple(self.ops)
        if any(not isinstance(op, FloatOpTypes) for op in ops):
            raise ValueError("FloatGraph contains an unsupported FloatOp type")
        inputs = tuple(self.inputs)
        outputs = tuple(self.outputs)
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("P0 FloatGraph requires exactly one input and one output")
        defined = set(constants) | set(inputs)
        for op in ops:
            if any(value not in defined for value in op.inputs):
                raise ValueError(f"{op.name} reads an undefined FloatValue")
            for value in op.outputs:
                if value not in values or value in defined:
                    raise ValueError(f"{op.name} has an invalid FloatValue output")
                defined.add(value)
        if outputs[0] not in defined:
            raise ValueError("FloatGraph output is not produced")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "constants", MappingProxyType(constants))
        object.__setattr__(self, "ops", ops)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)


__all__ = [
    "FloatAddOp",
    "FloatAveragePool1DOp",
    "FloatAveragePool2DOp",
    "FloatConcatOp",
    "FloatConv2DOp",
    "FloatConvTranspose2DOp",
    "FloatConv1DOp",
    "FloatDType",
    "FloatDepthwiseConv2DOp",
    "FloatFlattenOp",
    "FloatGraph",
    "FloatHardSigmoidOp",
    "FloatHardSwishOp",
    "FloatLayout",
    "FloatLinearOp",
    "FloatMaxPool2DOp",
    "FloatMaxPool1DOp",
    "FloatMulOp",
    "FloatPad2DOp",
    "FloatOp",
    "FloatReLU6Op",
    "FloatReLUOp",
    "FloatReshapeOp",
    "FloatReduceMeanOp",
    "FloatResizeNearest2DOp",
    "FloatResizeBilinear2DOp",
    "FloatSigmoidOp",
    "FloatSiLUOp",
    "FloatSliceOp",
    "FloatSoftmaxOp",
    "FloatValue",
    "FloatValueKind",
]
