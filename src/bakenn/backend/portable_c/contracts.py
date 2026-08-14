from __future__ import annotations

from dataclasses import dataclass, field
from functools import singledispatch
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from bakenn.errors import CompileError
from bakenn.plan import ExecutionPlan, ExecutionStep, Storage

if TYPE_CHECKING:
    from .selection import KernelSelection


@dataclass(frozen=True)
class ConstantEmission:
    """One immutable C constant declaration/definition pair."""

    symbol: str
    declaration: str
    definition: str
    size_bytes: int
    alignment: int = 1

    def __post_init__(self) -> None:
        if not self.symbol or not self.declaration or not self.definition:
            raise ValueError("C constant emissions must be complete")
        if self.size_bytes < 0:
            raise ValueError("C constant sizes cannot be negative")
        if (
            isinstance(self.alignment, bool)
            or not isinstance(self.alignment, int)
            or self.alignment <= 0
            or self.alignment & (self.alignment - 1)
        ):
            raise ValueError("C constant alignment must be a positive power of two")


@dataclass(frozen=True)
class KernelEmission:
    """One deduplicatable portable-C kernel supplied by an op family."""

    key: str
    declaration: str
    definition: str
    header_includes: tuple[str, ...] = ()
    source_includes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.key or not self.declaration or not self.definition:
            raise ValueError("C kernel emissions must be complete")
        object.__setattr__(self, "header_includes", tuple(self.header_includes))
        object.__setattr__(self, "source_includes", tuple(self.source_includes))


@dataclass(frozen=True)
class StepEmission:
    """All generated-C fragments owned by one execution-plan step."""

    constants: tuple[ConstantEmission, ...]
    kernels: tuple[KernelEmission, ...]
    call: str
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.call:
            raise ValueError("a step emission requires a model call")
        object.__setattr__(self, "constants", tuple(self.constants))
        object.__setattr__(self, "kernels", tuple(self.kernels))
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))


@dataclass(frozen=True)
class StepEmitContext:
    """Read-only services exposed to portable-C op-family emitters."""

    plan: ExecutionPlan
    symbol: str
    step_index: int
    constant_symbols: Mapping[str, str]
    selection: "KernelSelection | None" = None
    constant_overrides: Mapping[str, str] = field(default_factory=dict)
    backend_scratch_offset: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "constant_symbols", MappingProxyType(dict(self.constant_symbols)))
        object.__setattr__(
            self, "constant_overrides", MappingProxyType(dict(self.constant_overrides))
        )

    def pointer(self, name: str, *, mutable: bool) -> str:
        """Return the model-function expression for a plan tensor.

        Alias chains are resolved here so individual op families never need to
        know how physical storage is represented by the generic memory planner.
        """

        seen: set[str] = set()
        tensor_name = name
        while True:
            if tensor_name in seen:
                raise CompileError(f"cyclic plan alias while resolving {name}")
            seen.add(tensor_name)
            tensor = self.plan.tensors[tensor_name]
            if tensor.storage is not Storage.ALIAS:
                break
            assert tensor.alias_of is not None
            tensor_name = tensor.alias_of

        if tensor.storage is Storage.INPUT:
            return "input"
        if tensor.storage is Storage.OUTPUT:
            return "output"
        if tensor.storage is Storage.ARENA:
            cast = "int8_t *" if mutable else "const int8_t *"
            return f"({cast})(void *)(arena + {tensor.offset}u)"
        if tensor.storage is Storage.CONSTANT:
            if tensor_name in self.constant_overrides:
                return self.constant_overrides[tensor_name]
            try:
                return self.constant_symbols[tensor_name]
            except KeyError as error:
                raise CompileError(f"missing C symbol for constant tensor {tensor_name}") from error
        raise CompileError(
            f"portable C cannot resolve storage {tensor.storage.value} for tensor {name}"
        )

    @property
    def scratch_pointer(self) -> str:
        offset = (
            self.plan.scratch_offset
            if self.backend_scratch_offset is None
            else self.backend_scratch_offset
        )
        if offset is None:
            raise CompileError(f"step {self.step_index} requested scratch but the plan has none")
        return f"(void *)(arena + {offset}u)"


@singledispatch
def emit_step(step: object, context: StepEmitContext) -> StepEmission:
    """Emit one plan step, failing closed until its family module registers it."""

    del context
    raise CompileError(
        f"no portable C emitter registered for execution step type {type(step).__name__}"
    )


def checked_emit_step(step: ExecutionStep, context: StepEmitContext) -> StepEmission:
    """Typed wrapper used by the aggregator around the singledispatch hook."""

    emission = emit_step(step, context)
    if not isinstance(emission, StepEmission):
        raise CompileError(
            f"portable C emitter for {type(step).__name__} returned "
            f"{type(emission).__name__}, expected StepEmission"
        )
    return emission


__all__ = [
    "ConstantEmission",
    "KernelEmission",
    "StepEmitContext",
    "StepEmission",
    "checked_emit_step",
    "emit_step",
]
