from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from numbers import Integral
from types import MappingProxyType
from typing import ClassVar, Mapping, Protocol, runtime_checkable

import numpy as np

from bakenn.errors import CompileError
from bakenn.ir import TensorType


class Storage(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    CONSTANT = "constant"
    ARENA = "arena"
    ALIAS = "alias"
    SCRATCH = "scratch"


class AliasKind(str, Enum):
    VIEW = "view"
    INPLACE = "inplace"


@dataclass(frozen=True)
class BufferLifetime:
    """Half-open execution-step interval for one physical arena buffer."""

    birth: int
    death: int

    def __post_init__(self) -> None:
        if self.birth < 0 or self.death <= self.birth:
            raise ValueError("buffer lifetimes must be non-empty half-open intervals")

    def overlaps(self, other: "BufferLifetime") -> bool:
        return self.birth < other.death and other.birth < self.death


@dataclass(frozen=True)
class AliasSpec:
    """Declare that one step output shares physical storage with an input."""

    value: str
    target: str
    kind: AliasKind = AliasKind.VIEW
    byte_offset: int = 0

    def __post_init__(self) -> None:
        if not self.value or not self.target:
            raise ValueError("alias value and target names must be non-empty")
        if self.value == self.target:
            raise ValueError("an alias cannot target itself")
        if not isinstance(self.kind, AliasKind):
            raise ValueError("alias kind must use AliasKind")
        if isinstance(self.byte_offset, bool) or not isinstance(self.byte_offset, Integral):
            raise ValueError("alias byte_offset must be an integer")
        if self.byte_offset != 0:
            raise ValueError("P0 supports whole-buffer aliases at byte offset zero only")


@dataclass(frozen=True)
class PlanTensor:
    name: str
    tensor_type: TensorType
    storage: Storage
    offset: int | None = None
    alias_of: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("plan tensor name must be non-empty")
        if not isinstance(self.storage, Storage):
            raise ValueError("plan tensor storage must use Storage")
        if self.storage is Storage.ALIAS:
            if not self.alias_of or self.alias_of == self.name:
                raise ValueError("alias tensors require a distinct alias_of value")
        elif self.alias_of is not None:
            raise ValueError("only alias tensors may set alias_of")
        if self.storage is Storage.ARENA and self.offset is None:
            raise ValueError("arena tensors require a byte offset")
        if self.offset is not None and self.offset < 0:
            raise ValueError("plan tensor offsets cannot be negative")


@runtime_checkable
class ExecutionStep(Protocol):
    name: str
    kernel_kind: str
    arithmetic_profile: str

    @property
    def inputs(self) -> tuple[str, ...]: ...

    @property
    def outputs(self) -> tuple[str, ...]: ...

    @property
    def constants(self) -> tuple[str, ...]: ...

    @property
    def aliases(self) -> tuple[AliasSpec, ...]: ...

    @property
    def scratch_size(self) -> int: ...

    @property
    def scratch_alignment(self) -> int: ...


@dataclass(frozen=True)
class LinearStep:
    kernel_kind: ClassVar[str] = "linear_s8"

    name: str
    input: str
    weight: str
    bias: str
    output: str
    multipliers: tuple[int, ...]
    shifts: tuple[int, ...]
    activation_min: int
    activation_max: int
    accumulator_bounds: tuple[int, ...]
    arithmetic_profile: str = "bakenn.int8.linear.v1"

    def __post_init__(self) -> None:
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
class ExecutionPlan:
    name: str
    tensors: Mapping[str, PlanTensor]
    constants: Mapping[str, np.ndarray]
    steps: tuple[ExecutionStep, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    arena_size: int
    arena_alignment: int
    arithmetic_profile: str
    activation_arena_size: int = 0
    scratch_size: int = 0
    scratch_offset: int | None = None
    scratch_alignment: int = 1
    alias_groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    lifetimes: Mapping[str, BufferLifetime] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tensors = MappingProxyType(dict(self.tensors))
        constants: dict[str, np.ndarray] = {}
        for name, value in self.constants.items():
            frozen = np.array(value, copy=True, order="C")
            frozen.setflags(write=False)
            constants[name] = frozen
        steps = tuple(self.steps)
        for step in steps:
            if not is_dataclass(step) or not getattr(type(step), "__dataclass_params__", None).frozen:
                raise CompileError(f"execution step {type(step).__name__} must be an immutable dataclass")
            if not isinstance(getattr(step, "inputs", None), tuple):
                raise CompileError(f"execution step {step.name} must expose tuple inputs")
            if not isinstance(getattr(step, "outputs", None), tuple):
                raise CompileError(f"execution step {step.name} must expose tuple outputs")
        alias_groups = MappingProxyType(
            {name: tuple(members) for name, members in self.alias_groups.items()}
        )
        lifetimes = MappingProxyType(dict(self.lifetimes))
        if self.arena_alignment <= 0 or self.arena_alignment & (self.arena_alignment - 1):
            raise CompileError("arena alignment must be a positive power of two")
        if self.scratch_alignment <= 0 or self.scratch_alignment & (self.scratch_alignment - 1):
            raise CompileError("scratch alignment must be a positive power of two")
        if min(self.arena_size, self.activation_arena_size, self.scratch_size) < 0:
            raise CompileError("plan memory sizes cannot be negative")
        if self.activation_arena_size > self.arena_size:
            raise CompileError("activation arena cannot exceed total arena size")
        if self.scratch_size:
            if self.scratch_offset is None:
                raise CompileError("non-empty scratch requires an arena offset")
            if self.scratch_offset < self.activation_arena_size:
                raise CompileError("scratch overlaps activation arena")
            if self.scratch_offset % self.scratch_alignment:
                raise CompileError("scratch offset does not meet its alignment")
            if self.scratch_offset + self.scratch_size > self.arena_size:
                raise CompileError("scratch extends beyond the total arena")
        elif self.scratch_offset is not None:
            raise CompileError("zero-sized scratch must not have an arena offset")
        tensor_names = set(tensors)
        constant_names = set(constants)
        input_names = set(self.inputs)
        output_names = set(self.outputs)
        if len(self.inputs) != 1 or len(self.outputs) != 1:
            raise CompileError("P0 execution plans require exactly one input and one output")
        if input_names - tensor_names or output_names - tensor_names:
            raise CompileError("execution-plan I/O references an unknown tensor")
        if constant_names - tensor_names:
            raise CompileError("execution-plan constants require typed plan tensors")

        def physical_storage(name: str) -> Storage:
            seen: set[str] = set()
            current = name
            while tensors[current].storage is Storage.ALIAS:
                if current in seen:
                    raise CompileError(f"plan alias cycle includes {current}")
                seen.add(current)
                target = tensors[current].alias_of
                if target not in tensor_names:
                    raise CompileError(f"plan alias {current} targets an unknown tensor")
                assert target is not None
                current = target
            return tensors[current].storage

        for name in input_names:
            if physical_storage(name) is not Storage.INPUT:
                raise CompileError(f"plan input {name} must use input storage")
        for name in output_names:
            if physical_storage(name) is not Storage.OUTPUT:
                raise CompileError(f"plan output {name} must use output storage")
        for name in constant_names:
            if physical_storage(name) is not Storage.CONSTANT:
                raise CompileError(f"plan constant {name} must use constant storage")
        for name, tensor in tensors.items():
            if tensor.storage is Storage.ALIAS and tensor.alias_of not in tensor_names:
                raise CompileError(f"plan alias {name} targets an unknown tensor")
        for step in steps:
            unknown = set((*step.inputs, *step.outputs, *step.constants)) - tensor_names
            if unknown:
                raise CompileError(
                    f"execution step {step.name} references unknown tensors: {sorted(unknown)}"
                )
        arena_roots = {
            name for name, tensor in tensors.items() if tensor.storage is Storage.ARENA
        }
        if lifetimes and set(lifetimes) != arena_roots:
            raise CompileError("execution-plan lifetimes must exactly describe arena roots")
        for name, lifetime in lifetimes.items():
            if not isinstance(lifetime, BufferLifetime):
                raise CompileError(f"execution-plan lifetime for {name} has an invalid type")
            if lifetime.death > len(steps):
                raise CompileError(f"execution-plan lifetime for {name} exceeds the step schedule")
        object.__setattr__(self, "tensors", tensors)
        object.__setattr__(self, "constants", MappingProxyType(constants))
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "alias_groups", alias_groups)
        object.__setattr__(self, "lifetimes", lifetimes)


__all__ = [
    "AliasKind",
    "AliasSpec",
    "BufferLifetime",
    "ExecutionPlan",
    "ExecutionStep",
    "LinearStep",
    "PlanTensor",
    "Storage",
]
