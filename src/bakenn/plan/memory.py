from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Mapping, Sequence

from bakenn.errors import CompileError
from bakenn.ir import TensorType
from bakenn.ir.types import TARGET_SIZE_MAX
from .types import AliasKind, AliasSpec, BufferLifetime, ExecutionStep, PlanTensor, Storage


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _power_of_two(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise CompileError(f"{description} must be an integer")
    normalized = int(value)
    if normalized <= 0 or normalized & (normalized - 1):
        raise CompileError(f"{description} must be a positive power of two")
    return normalized


@dataclass(frozen=True)
class MemoryLayout:
    tensors: Mapping[str, PlanTensor]
    alias_groups: Mapping[str, tuple[str, ...]]
    lifetimes: Mapping[str, BufferLifetime]
    activation_arena_size: int
    scratch_size: int
    scratch_offset: int | None
    scratch_alignment: int
    arena_size: int
    arena_alignment: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))
        object.__setattr__(
            self,
            "alias_groups",
            MappingProxyType({name: tuple(members) for name, members in self.alias_groups.items()}),
        )
        object.__setattr__(self, "lifetimes", MappingProxyType(dict(self.lifetimes)))
        validate_memory_layout(self)


def _step_values(step: ExecutionStep) -> tuple[tuple[str, ...], tuple[str, ...]]:
    inputs = getattr(step, "inputs", None)
    outputs = getattr(step, "outputs", None)
    if not isinstance(inputs, tuple) or not all(isinstance(name, str) and name for name in inputs):
        raise CompileError(f"step {getattr(step, 'name', '<unnamed>')} has invalid inputs")
    if not isinstance(outputs, tuple) or not outputs or not all(
        isinstance(name, str) and name for name in outputs
    ):
        raise CompileError(f"step {getattr(step, 'name', '<unnamed>')} has invalid outputs")
    return inputs, outputs


def _collect_aliases(steps: Sequence[ExecutionStep]) -> tuple[AliasSpec, ...]:
    result: list[AliasSpec] = []
    for step in steps:
        aliases = getattr(step, "aliases", ())
        if not isinstance(aliases, tuple) or not all(isinstance(alias, AliasSpec) for alias in aliases):
            raise CompileError(f"step {step.name} has invalid alias declarations")
        result.extend(aliases)
    return tuple(result)


def plan_memory(
    *,
    values: Mapping[str, TensorType],
    constants: Sequence[str] | set[str],
    inputs: Sequence[str],
    outputs: Sequence[str],
    steps: Sequence[ExecutionStep],
    arena_alignment: int = 16,
) -> MemoryLayout:
    """Plan activation aliases, half-open liveness, and reusable step scratch."""

    base_alignment = _power_of_two(arena_alignment, "arena alignment")
    if base_alignment > TARGET_SIZE_MAX:
        raise CompileError("arena alignment exceeds the 32-bit target ABI")
    value_names = set(values)
    constant_names = set(constants)
    input_names = set(inputs)
    output_names = set(outputs)
    for group_name, names in (
        ("constant", constant_names),
        ("input", input_names),
        ("output", output_names),
    ):
        unknown = names - value_names
        if unknown:
            raise CompileError(f"unknown {group_name} values in memory plan: {sorted(unknown)}")

    births: dict[str, int] = {}
    last_uses: dict[str, int] = {}
    producer_step: dict[str, int] = {}
    step_inputs: list[tuple[str, ...]] = []
    step_outputs: list[tuple[str, ...]] = []
    normalized_steps = tuple(steps)
    for index, step in enumerate(normalized_steps):
        op_inputs, op_outputs = _step_values(step)
        step_inputs.append(op_inputs)
        step_outputs.append(op_outputs)
        for name in op_inputs:
            if name not in value_names:
                raise CompileError(f"step {step.name} reads unknown value {name}")
            last_uses[name] = index
        for name in op_outputs:
            if name not in value_names:
                raise CompileError(f"step {step.name} writes unknown value {name}")
            if name in producer_step or name in input_names or name in constant_names:
                raise CompileError(f"memory plan found multiple producers for {name}")
            births[name] = index
            producer_step[name] = index

    alias_map: dict[str, AliasSpec] = {}
    for alias in _collect_aliases(normalized_steps):
        if alias.value not in value_names or alias.target not in value_names:
            raise CompileError(f"alias {alias.value}->{alias.target} references an unknown value")
        if alias.value in alias_map:
            raise CompileError(f"multiple alias declarations for {alias.value}")
        produced_at = producer_step.get(alias.value)
        if produced_at is None:
            raise CompileError(f"alias value {alias.value} is not produced by a step")
        if alias.target not in step_inputs[produced_at]:
            raise CompileError(
                f"alias {alias.value}->{alias.target} must share storage with an input of its producer"
            )
        value_type = values[alias.value]
        target_type = values[alias.target]
        if value_type.nbytes != target_type.nbytes or value_type.dtype is not target_type.dtype:
            raise CompileError(f"alias {alias.value}->{alias.target} has incompatible storage size or dtype")
        if alias.kind is AliasKind.VIEW and value_type.qparams != target_type.qparams:
            raise CompileError(f"view alias {alias.value}->{alias.target} must preserve qparams")
        alias_map[alias.value] = alias

    def root(name: str) -> str:
        pending: set[str] = set()
        current = name
        while current in alias_map:
            if current in pending:
                raise CompileError(f"alias cycle includes {current}")
            pending.add(current)
            current = alias_map[current].target
        return current

    groups: dict[str, list[str]] = {}
    for name in values:
        groups.setdefault(root(name), []).append(name)

    for alias in alias_map.values():
        if alias.kind is AliasKind.INPLACE:
            step_index = producer_step[alias.value]
            target_group = groups[root(alias.target)]
            live_aliases = [
                member
                for member in target_group
                if member != alias.value
                and births.get(member, -1) <= step_index
                and (
                    last_uses.get(member, births.get(member, -1)) > step_index
                    or member in output_names
                )
            ]
            if live_aliases:
                raise CompileError(
                    f"unsafe in-place alias {alias.value}->{alias.target}: alias group has a later "
                    f"consumer or output member {sorted(live_aliases)}"
                )
            if input_names.intersection(target_group) or constant_names.intersection(target_group):
                raise CompileError(
                    f"unsafe in-place alias {alias.value}->{alias.target}: caller/constant storage is read-only"
                )

    group_storage: dict[str, Storage] = {}
    for group_root, members in groups.items():
        member_set = set(members)
        owns_input = bool(member_set & input_names)
        owns_output = bool(member_set & output_names)
        owns_constant = bool(member_set & constant_names)
        if sum((owns_input, owns_output, owns_constant)) > 1:
            raise CompileError(
                f"unsafe alias group {members}: caller input, output, and constants may not overlap"
            )
        if owns_input:
            group_storage[group_root] = Storage.INPUT
        elif owns_output:
            group_storage[group_root] = Storage.OUTPUT
        elif owns_constant:
            group_storage[group_root] = Storage.CONSTANT
        else:
            group_storage[group_root] = Storage.ARENA

    lifetimes: dict[str, BufferLifetime] = {}
    for group_root, members in groups.items():
        if group_storage[group_root] is not Storage.ARENA:
            continue
        member_births = [births[name] for name in members if name in births]
        if not member_births:
            raise CompileError(f"arena alias group {members} has no producer")
        birth = min(member_births)
        last_use = max((last_uses.get(name, births.get(name, birth)) for name in members), default=birth)
        lifetimes[group_root] = BufferLifetime(birth, max(birth, last_use) + 1)

    active: list[tuple[str, BufferLifetime, int, int]] = []
    offsets: dict[str, int] = {}
    peak = 0
    candidates = sorted(lifetimes, key=lambda name: (lifetimes[name].birth, name))
    for group_root in candidates:
        lifetime = lifetimes[group_root]
        active = [entry for entry in active if entry[1].death > lifetime.birth]
        size = max(values[name].nbytes for name in groups[group_root])
        occupied = sorted((offset, offset + used_size) for _, _, offset, used_size in active)
        offset = 0
        for start, end in occupied:
            aligned = _align_up(offset, base_alignment)
            if aligned + size <= start:
                offset = aligned
                break
            offset = max(offset, end)
        else:
            offset = _align_up(offset, base_alignment)
        offsets[group_root] = offset
        active.append((group_root, lifetime, offset, size))
        peak = max(peak, offset + size)

    activation_size = _align_up(peak, base_alignment) if peak else 0
    scratch_size = 0
    scratch_alignment = 1
    for step in normalized_steps:
        size = getattr(step, "scratch_size", None)
        alignment = getattr(step, "scratch_alignment", None)
        if isinstance(size, bool) or not isinstance(size, Integral) or size < 0:
            raise CompileError(f"step {step.name} scratch_size must be a non-negative integer")
        step_alignment = _power_of_two(alignment, f"step {step.name} scratch alignment")
        scratch_size = max(scratch_size, int(size))
        scratch_alignment = max(scratch_alignment, step_alignment)
    final_alignment = max(base_alignment, scratch_alignment)
    scratch_offset = _align_up(activation_size, scratch_alignment) if scratch_size else None
    arena_size = (
        _align_up(scratch_offset + scratch_size, final_alignment)
        if scratch_offset is not None
        else _align_up(activation_size, final_alignment)
    )
    if max(activation_size, scratch_size, scratch_offset or 0, arena_size) > TARGET_SIZE_MAX:
        raise CompileError("planned arena or scratch storage exceeds the 32-bit target ABI")

    tensors: dict[str, PlanTensor] = {}
    for name, tensor_type in values.items():
        group_root = root(name)
        if name in alias_map:
            physical_offset = offsets.get(group_root)
            tensors[name] = PlanTensor(
                name,
                tensor_type,
                Storage.ALIAS,
                physical_offset,
                alias_of=alias_map[name].target,
            )
            continue
        storage = group_storage[group_root]
        tensors[name] = PlanTensor(name, tensor_type, storage, offsets.get(group_root))

    return MemoryLayout(
        tensors=tensors,
        alias_groups={name: tuple(members) for name, members in groups.items()},
        lifetimes=lifetimes,
        activation_arena_size=activation_size,
        scratch_size=scratch_size,
        scratch_offset=scratch_offset,
        scratch_alignment=scratch_alignment,
        arena_size=arena_size,
        arena_alignment=final_alignment,
    )


def validate_memory_layout(layout: MemoryLayout) -> None:
    if layout.arena_alignment <= 0 or layout.arena_alignment & (layout.arena_alignment - 1):
        raise CompileError("arena alignment must be a positive power of two")
    if layout.activation_arena_size > layout.arena_size:
        raise CompileError("activation arena exceeds total arena")
    arena_groups: list[tuple[str, BufferLifetime, int, int]] = []
    for group_root, lifetime in layout.lifetimes.items():
        tensor = layout.tensors[group_root]
        if tensor.storage is not Storage.ARENA or tensor.offset is None:
            raise CompileError(f"lifetime for non-arena group {group_root}")
        members = layout.alias_groups[group_root]
        size = max(layout.tensors[name].tensor_type.nbytes for name in members)
        if tensor.offset + size > layout.activation_arena_size:
            raise CompileError(f"arena buffer {group_root} extends beyond activation allocation")
        arena_groups.append((group_root, lifetime, tensor.offset, size))
    for index, (left_name, left_life, left_offset, left_size) in enumerate(arena_groups):
        for right_name, right_life, right_offset, right_size in arena_groups[index + 1 :]:
            memory_overlaps = left_offset < right_offset + right_size and right_offset < left_offset + left_size
            if memory_overlaps and left_life.overlaps(right_life):
                raise CompileError(
                    f"live arena buffers overlap: {left_name} and {right_name}"
                )
    if layout.scratch_size:
        if layout.scratch_offset is None or layout.scratch_offset < layout.activation_arena_size:
            raise CompileError("scratch overlaps activation arena")
        if layout.scratch_offset % layout.scratch_alignment:
            raise CompileError("scratch offset is misaligned")
        if layout.scratch_offset + layout.scratch_size > layout.arena_size:
            raise CompileError("scratch exceeds total arena")
    elif layout.scratch_offset is not None:
        raise CompileError("zero scratch must not reserve an offset")


__all__ = ["BufferLifetime", "MemoryLayout", "plan_memory", "validate_memory_layout"]
