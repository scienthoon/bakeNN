from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import ClassVar

from bakenn.plan.types import AliasKind, AliasSpec
from bakenn.quantization.fixedpoint import INT32_MAX


def _integer(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    normalized = int(value)
    if not minimum <= normalized <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return normalized


def _validate_q31(instance: object, prefix: str) -> None:
    multiplier_name = f"{prefix}_multiplier"
    shift_name = f"{prefix}_shift"
    multiplier = _integer(
        getattr(instance, multiplier_name), multiplier_name, minimum=0, maximum=INT32_MAX
    )
    shift = _integer(getattr(instance, shift_name), shift_name, minimum=-31, maximum=30)
    if multiplier == 0:
        if shift != 0:
            raise ValueError(f"{shift_name} must be zero for an underflow multiplier")
    elif multiplier < 1 << 30:
        raise ValueError(f"{multiplier_name} must be zero or a normalized Q31 multiplier")
    object.__setattr__(instance, multiplier_name, multiplier)
    object.__setattr__(instance, shift_name, shift)


def _validate_clamp(instance: object) -> None:
    minimum = _integer(
        getattr(instance, "activation_min"), "activation_min", minimum=-128, maximum=127
    )
    maximum = _integer(
        getattr(instance, "activation_max"), "activation_max", minimum=-128, maximum=127
    )
    if minimum > maximum:
        raise ValueError("activation_min must not exceed activation_max")
    object.__setattr__(instance, "activation_min", minimum)
    object.__setattr__(instance, "activation_max", maximum)


def _validate_bound(instance: object, name: str) -> None:
    object.__setattr__(
        instance,
        name,
        _integer(getattr(instance, name), name, minimum=0, maximum=INT32_MAX),
    )


@dataclass(frozen=True)
class AddStep:
    kernel_kind: ClassVar[str] = "add_s8"
    left_shift: ClassVar[int] = 20

    name: str
    input_a: str
    input_b: str
    output: str
    input_a_multiplier: int
    input_a_shift: int
    input_b_multiplier: int
    input_b_shift: int
    output_multiplier: int
    output_shift: int
    activation_min: int
    activation_max: int
    input_a_centered_bound: int
    input_b_centered_bound: int
    input_a_shifted_bound: int
    input_b_shifted_bound: int
    input_a_pre_high_mul_bound: int
    input_b_pre_high_mul_bound: int
    input_a_scaled_bound: int
    input_b_scaled_bound: int
    sum_bound: int
    output_pre_high_mul_bound: int
    arithmetic_profile: str = "bakenn.int8.add.v1"

    def __post_init__(self) -> None:
        for prefix in ("input_a", "input_b", "output"):
            _validate_q31(self, prefix)
        _validate_clamp(self)
        for name in (
            "input_a_centered_bound",
            "input_b_centered_bound",
            "input_a_shifted_bound",
            "input_b_shifted_bound",
            "input_a_pre_high_mul_bound",
            "input_b_pre_high_mul_bound",
            "input_a_scaled_bound",
            "input_b_scaled_bound",
            "sum_bound",
            "output_pre_high_mul_bound",
        ):
            _validate_bound(self, name)

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input_a, self.input_b)

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


@dataclass(frozen=True)
class MulStep:
    kernel_kind: ClassVar[str] = "mul_s8"

    name: str
    input_a: str
    input_b: str
    output: str
    output_multiplier: int
    output_shift: int
    activation_min: int
    activation_max: int
    input_a_centered_bound: int
    input_b_centered_bound: int
    product_bound: int
    requantize_pre_high_mul_bound: int
    arithmetic_profile: str = "bakenn.int8.mul.v1"

    def __post_init__(self) -> None:
        _validate_q31(self, "output")
        _validate_clamp(self)
        for name in (
            "input_a_centered_bound",
            "input_b_centered_bound",
            "product_bound",
            "requantize_pre_high_mul_bound",
        ):
            _validate_bound(self, name)

    @property
    def inputs(self) -> tuple[str, ...]:
        return (self.input_a, self.input_b)

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


@dataclass(frozen=True)
class ClampStep:
    kernel_kind: ClassVar[str] = "clamp_s8"

    name: str
    input: str
    output: str
    activation_min: int
    activation_max: int
    inplace: bool = False
    arithmetic_profile: str = "bakenn.int8.clamp.v1"

    def __post_init__(self) -> None:
        _validate_clamp(self)
        if not isinstance(self.inplace, bool):
            raise ValueError("inplace must be a boolean")

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
        if not self.inplace:
            return ()
        return (AliasSpec(self.output, self.input, AliasKind.INPLACE),)

    @property
    def scratch_size(self) -> int:
        return 0

    @property
    def scratch_alignment(self) -> int:
        return 1


@dataclass(frozen=True)
class RequantizeStep:
    kernel_kind: ClassVar[str] = "requantize_s8"

    name: str
    input: str
    output: str
    multiplier: int
    shift: int
    centered_bound: int
    requantize_pre_high_mul_bound: int
    inplace: bool = False
    activation_min: int = -128
    activation_max: int = 127
    arithmetic_profile: str = "bakenn.int8.requantize.v1"

    def __post_init__(self) -> None:
        multiplier = _integer(self.multiplier, "multiplier", minimum=0, maximum=INT32_MAX)
        shift = _integer(self.shift, "shift", minimum=-31, maximum=30)
        if multiplier == 0:
            if shift != 0:
                raise ValueError("shift must be zero for an underflow multiplier")
        elif multiplier < 1 << 30:
            raise ValueError("multiplier must be zero or a normalized Q31 multiplier")
        object.__setattr__(self, "multiplier", multiplier)
        object.__setattr__(self, "shift", shift)
        _validate_bound(self, "centered_bound")
        _validate_bound(self, "requantize_pre_high_mul_bound")
        if not isinstance(self.inplace, bool):
            raise ValueError("inplace must be a boolean")
        _validate_clamp(self)

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
        if not self.inplace:
            return ()
        return (AliasSpec(self.output, self.input, AliasKind.INPLACE),)

    @property
    def scratch_size(self) -> int:
        return 0

    @property
    def scratch_alignment(self) -> int:
        return 1


__all__ = ["AddStep", "ClampStep", "MulStep", "RequantizeStep"]
