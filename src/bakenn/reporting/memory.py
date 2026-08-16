from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import TYPE_CHECKING

from bakenn._version import VERSION
from bakenn.errors import CompileError
from bakenn.plan import ExecutionPlan, Storage

if TYPE_CHECKING:
    from bakenn.backend.portable_c.selection import CBackendPlan


_POST_LINK_REASON = (
    "requires a final target ELF/map built with the production compiler, flags, and linker script"
)
_FULL_SRAM_REASON = (
    "requires the final firmware and includes caller I/O, application globals, RTOS/interrupt "
    "state, and stacks"
)
_STACK_REASON = "requires target stack analysis or a physical-board high-water measurement"


def _format_bytes(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    return f"{value / 1024.0:.1f} KiB ({value} B)"


@dataclass(frozen=True)
class ArenaBufferReport:
    """One physical activation allocation and its half-open execution lifetime."""

    name: str
    members: tuple[str, ...]
    offset: int
    size_bytes: int
    birth_step: int
    death_step_exclusive: int

    def json_data(self) -> dict[str, object]:
        return {
            "name": self.name,
            "members": list(self.members),
            "offset": self.offset,
            "size_bytes": self.size_bytes,
            "lifetime": {
                "birth_step": self.birth_step,
                "death_step_exclusive": self.death_step_exclusive,
                "last_live_step": self.death_step_exclusive - 1,
            },
        }


@dataclass(frozen=True)
class ReuseRegion:
    """A byte range occupied by different activation buffers at disjoint times."""

    offset: int
    size_bytes: int
    buffers: tuple[str, ...]

    def json_data(self) -> dict[str, object]:
        return {
            "offset": self.offset,
            "size_bytes": self.size_bytes,
            "buffers": list(self.buffers),
        }


@dataclass(frozen=True)
class KernelStepMemory:
    index: int
    name: str
    kernel_id: str
    optimized: bool
    live_buffers: tuple[str, ...]
    live_activation_payload_bytes: int
    scratch_bytes: int
    working_payload_bytes: int

    def json_data(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "kernel_id": self.kernel_id,
            "optimized": self.optimized,
            "live_buffers": list(self.live_buffers),
            "live_activation_payload_bytes": self.live_activation_payload_bytes,
            "scratch_bytes": self.scratch_bytes,
            "working_payload_bytes": self.working_payload_bytes,
        }


@dataclass(frozen=True)
class MemoryReport:
    """Deterministic model-memory facts known at AOT compilation time.

    This deliberately does not pretend that constant payload is final Flash or
    that the model arena is whole-firmware peak SRAM.  Those values require a
    target link or a physical measurement and remain explicit unknowns here.
    """

    model: str
    target: str
    semantic_constant_bytes: int
    emitted_constant_payload_bytes: int
    input_bytes: int
    output_bytes: int
    activation_arena_bytes: int
    scratch_bytes: int
    scratch_offset: int | None
    arena_bytes: int
    arena_alignment: int
    target_flash_budget_bytes: int | None
    target_sram_budget_bytes: int | None
    buffers: tuple[ArenaBufferReport, ...]
    reuse_regions: tuple[ReuseRegion, ...]
    steps: tuple[KernelStepMemory, ...]
    peak_working_payload_bytes: int
    peak_step_indices: tuple[int, ...]

    @property
    def caller_io_bytes(self) -> int:
        return self.input_bytes + self.output_bytes

    @property
    def generated_model_heap_calls(self) -> int:
        return 0

    def json_data(self) -> dict[str, object]:
        peak_steps = [self.steps[index].name for index in self.peak_step_indices]
        flash_headroom = (
            None
            if self.target_flash_budget_bytes is None
            else self.target_flash_budget_bytes - self.emitted_constant_payload_bytes
        )
        arena_headroom = (
            None
            if self.target_sram_budget_bytes is None
            else self.target_sram_budget_bytes - self.arena_bytes
        )
        return {
            "schema_version": 1,
            "compiler_version": VERSION,
            "model": self.model,
            "target": self.target,
            "scope": {
                "generated_model_only": True,
                "arena_excludes_caller_io": True,
                "constant_payload_is_not_final_flash": True,
            },
            "compile_time": {
                "semantic_constant_bytes": self.semantic_constant_bytes,
                "emitted_constant_payload_bytes": self.emitted_constant_payload_bytes,
                "input_bytes": self.input_bytes,
                "output_bytes": self.output_bytes,
                "caller_io_bytes": self.caller_io_bytes,
                "activation_arena_bytes": self.activation_arena_bytes,
                "scratch_bytes": self.scratch_bytes,
                "scratch_offset": self.scratch_offset,
                "arena_bytes": self.arena_bytes,
                "arena_alignment": self.arena_alignment,
                "generated_model_heap_calls": self.generated_model_heap_calls,
            },
            "target_budgets": {
                "flash_bytes": self.target_flash_budget_bytes,
                "constant_payload_headroom_bytes": flash_headroom,
                "constant_payload_headroom_scope": "generated code and initialized data excluded",
                "sram_bytes": self.target_sram_budget_bytes,
                "arena_headroom_bytes": arena_headroom,
                "arena_headroom_scope": "caller I/O, application globals, and stacks excluded",
            },
            "peak_live_working_payload": {
                "bytes": self.peak_working_payload_bytes,
                "step_indices": list(self.peak_step_indices),
                "step_names": peak_steps,
                "definition": "live activation payload plus scratch used by the selected step",
            },
            "arena_buffers": [buffer.json_data() for buffer in self.buffers],
            "reuse_regions": [region.json_data() for region in self.reuse_regions],
            "steps": [step.json_data() for step in self.steps],
            "not_measured_at_aot": {
                "final_flash_load_bytes": {"value": None, "reason": _POST_LINK_REASON},
                "full_firmware_peak_sram_bytes": {"value": None, "reason": _FULL_SRAM_REASON},
                "peak_stack_bytes": {"value": None, "reason": _STACK_REASON},
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.json_data(), indent=2, sort_keys=True) + "\n"

    def to_text(self) -> str:
        lines = [
            "BakeNN Memory Report",
            f"Model: {self.model}",
            f"Target: {self.target}",
            "Scope: generated model only; caller I/O and application memory are separate.",
            "",
            "Compile-time exact",
            f"  Semantic constants       {_format_bytes(self.semantic_constant_bytes)}",
            f"  Emitted constant payload {_format_bytes(self.emitted_constant_payload_bytes)}",
            f"  Activation arena         {_format_bytes(self.activation_arena_bytes)}",
            f"  Kernel scratch           {_format_bytes(self.scratch_bytes)}",
            f"  Total model arena        {_format_bytes(self.arena_bytes)}",
            f"  Arena alignment          {self.arena_alignment} B",
            f"  Caller input/output      {_format_bytes(self.caller_io_bytes)} "
            f"({self.input_bytes} B + {self.output_bytes} B; not in arena)",
            "  Generated-model heap     0 calls",
        ]
        if self.target_flash_budget_bytes is not None or self.target_sram_budget_bytes is not None:
            lines.extend(("", "Declared target budgets (lower-bound checks)"))
            if self.target_flash_budget_bytes is not None:
                headroom = self.target_flash_budget_bytes - self.emitted_constant_payload_bytes
                lines.append(
                    f"  Flash constant headroom  {_format_bytes(headroom)} "
                    "(code/data excluded)"
                )
            if self.target_sram_budget_bytes is not None:
                headroom = self.target_sram_budget_bytes - self.arena_bytes
                lines.append(
                    f"  SRAM arena headroom      {_format_bytes(headroom)} "
                    "(caller I/O/app/stack excluded)"
                )

        lines.extend(("", "Peak live working payload"))
        if self.peak_step_indices:
            names = ", ".join(self.steps[index].name for index in self.peak_step_indices)
            lines.append(f"  {_format_bytes(self.peak_working_payload_bytes)} at {names}")
            first = self.steps[self.peak_step_indices[0]]
            for name in first.live_buffers:
                buffer = next(item for item in self.buffers if item.name == name)
                lines.append(
                    f"    {buffer.name}: offset {buffer.offset}, {_format_bytes(buffer.size_bytes)}, "
                    f"steps {buffer.birth_step}..{buffer.death_step_exclusive - 1}"
                )
            if first.scratch_bytes:
                lines.append(f"    selected-kernel scratch: {_format_bytes(first.scratch_bytes)}")
        else:
            lines.append("  0 B (no arena activation or scratch allocation)")

        lines.extend(("", "Reused arena regions"))
        if self.reuse_regions:
            for region in self.reuse_regions:
                lines.append(
                    f"  offset {region.offset}, {_format_bytes(region.size_bytes)}: "
                    + " -> ".join(region.buffers)
                )
        else:
            lines.append("  none")

        lines.extend(("", "Selected kernels"))
        for step in self.steps:
            marker = "optimized" if step.optimized else "portable"
            lines.append(
                f"  [{step.index}] {step.name}: {step.kernel_id} ({marker}), "
                f"scratch {_format_bytes(step.scratch_bytes)}"
            )

        lines.extend(
            (
                "",
                "Not measured at AOT compile time",
                f"  Final Flash load: {_POST_LINK_REASON}.",
                f"  Full firmware peak SRAM: {_FULL_SRAM_REASON}.",
                f"  Peak stack: {_STACK_REASON}.",
                "",
            )
        )
        return "\n".join(lines)

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.write_text(self.to_json(), encoding="utf-8")
        return output

    def write_text(self, path: str | Path) -> Path:
        output = Path(path)
        output.write_text(self.to_text(), encoding="utf-8")
        return output


def _physical_groups(plan: ExecutionPlan) -> dict[str, tuple[str, ...]]:
    def root(name: str) -> str:
        seen: set[str] = set()
        current = name
        while plan.tensors[current].storage is Storage.ALIAS:
            if current in seen:
                raise ValueError(f"execution plan alias cycle includes {current}")
            seen.add(current)
            target = plan.tensors[current].alias_of
            if target is None:
                raise ValueError(f"execution plan alias {current} has no target")
            current = target
        return current

    groups: dict[str, list[str]] = {}
    for name in plan.tensors:
        groups.setdefault(root(name), []).append(name)
    return {name: tuple(members) for name, members in groups.items()}


def _arena_buffers(plan: ExecutionPlan) -> tuple[ArenaBufferReport, ...]:
    births: dict[str, int] = {}
    last_uses: dict[str, int] = {}
    for index, step in enumerate(plan.steps):
        for name in step.inputs:
            last_uses[name] = index
        for name in step.outputs:
            births[name] = index

    reports: list[ArenaBufferReport] = []
    for root, members in _physical_groups(plan).items():
        tensor = plan.tensors[root]
        if tensor.storage is not Storage.ARENA:
            continue
        member_births = [births[name] for name in members if name in births]
        if not member_births or tensor.offset is None:
            raise ValueError(f"arena group {root} lacks a producer or physical offset")
        birth = min(member_births)
        last_use = max(
            (last_uses.get(name, births.get(name, birth)) for name in members),
            default=birth,
        )
        planned_lifetime = plan.lifetimes.get(root)
        if planned_lifetime is not None:
            birth = planned_lifetime.birth
            death = planned_lifetime.death
        else:
            death = max(birth, last_use) + 1
        reports.append(
            ArenaBufferReport(
                name=root,
                members=members,
                offset=tensor.offset,
                size_bytes=max(plan.tensors[name].tensor_type.nbytes for name in members),
                birth_step=birth,
                death_step_exclusive=death,
            )
        )
    return tuple(sorted(reports, key=lambda item: (item.offset, item.birth_step, item.name)))


def _reuse_regions(buffers: tuple[ArenaBufferReport, ...]) -> tuple[ReuseRegion, ...]:
    by_name = {item.name: item for item in buffers}
    boundaries = sorted(
        {boundary for item in buffers for boundary in (item.offset, item.offset + item.size_bytes)}
    )
    regions: list[ReuseRegion] = []
    for start, end in zip(boundaries, boundaries[1:]):
        occupants = tuple(
            item.name
            for item in buffers
            if item.offset <= start and end <= item.offset + item.size_bytes
        )
        if len(occupants) < 2:
            continue
        for left_name, right_name in combinations(occupants, 2):
            left = by_name[left_name]
            right = by_name[right_name]
            if (
                left.birth_step < right.death_step_exclusive
                and right.birth_step < left.death_step_exclusive
            ):
                raise CompileError(
                    f"memory report found simultaneously live arena buffers sharing bytes: "
                    f"{left.name} and {right.name}"
                )
        if (
            regions
            and regions[-1].offset + regions[-1].size_bytes == start
            and regions[-1].buffers == occupants
        ):
            previous = regions[-1]
            regions[-1] = ReuseRegion(
                previous.offset,
                previous.size_bytes + end - start,
                occupants,
            )
        else:
            regions.append(ReuseRegion(start, end - start, occupants))
    return tuple(regions)


def build_memory_report(
    plan: ExecutionPlan,
    backend_plan: "CBackendPlan",
    *,
    emitted_constant_payload_bytes: int,
) -> MemoryReport:
    """Build the report from one immutable semantic and selected backend plan."""

    if backend_plan.execution_plan is not plan:
        raise ValueError("memory report requires the backend plan for the same execution plan")
    if emitted_constant_payload_bytes < 0:
        raise ValueError("emitted constant payload cannot be negative")
    buffers = _arena_buffers(plan)
    buffer_by_name = {item.name: item for item in buffers}
    step_reports: list[KernelStepMemory] = []
    for index, (step, selection) in enumerate(zip(plan.steps, backend_plan.selections)):
        live = tuple(
            item.name
            for item in buffers
            if item.birth_step <= index < item.death_step_exclusive
        )
        live_payload = sum(buffer_by_name[name].size_bytes for name in live)
        scratch = max(int(step.scratch_size), selection.scratch_size)
        step_reports.append(
            KernelStepMemory(
                index=index,
                name=step.name,
                kernel_id=selection.kernel_id,
                optimized=selection.optimized,
                live_buffers=live,
                live_activation_payload_bytes=live_payload,
                scratch_bytes=scratch,
                working_payload_bytes=live_payload + scratch,
            )
        )
    steps = tuple(step_reports)
    peak = max((step.working_payload_bytes for step in steps), default=0)
    peak_indices = tuple(
        step.index for step in steps if step.working_payload_bytes == peak and peak
    )
    target = backend_plan.options.target
    input_bytes = plan.tensors[plan.inputs[0]].tensor_type.nbytes
    output_bytes = plan.tensors[plan.outputs[0]].tensor_type.nbytes
    return MemoryReport(
        model=plan.name,
        target=target.target_id,
        semantic_constant_bytes=sum(int(value.nbytes) for value in plan.constants.values()),
        emitted_constant_payload_bytes=emitted_constant_payload_bytes,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        activation_arena_bytes=backend_plan.activation_arena_size,
        scratch_bytes=backend_plan.scratch_size,
        scratch_offset=backend_plan.scratch_offset,
        arena_bytes=backend_plan.arena_size,
        arena_alignment=backend_plan.arena_alignment,
        target_flash_budget_bytes=target.flash_bytes,
        target_sram_budget_bytes=target.sram_bytes,
        buffers=buffers,
        reuse_regions=_reuse_regions(buffers),
        steps=steps,
        peak_working_payload_bytes=peak,
        peak_step_indices=peak_indices,
    )


__all__ = [
    "ArenaBufferReport",
    "KernelStepMemory",
    "MemoryReport",
    "ReuseRegion",
    "build_memory_report",
]
