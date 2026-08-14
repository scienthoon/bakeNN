from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping


_TARGET_SIZE_MAX = (1 << 32) - 1


class TargetArchitecture(str, Enum):
    """Instruction-set family used by a generated target artifact."""

    PORTABLE = "portable"
    ARM = "arm"
    RISCV = "riscv"
    XTENSA = "xtensa"


@dataclass(frozen=True)
class KernelCostMeasurement:
    """One physical-target measurement; never an inferred performance value."""

    kernel_id: str
    workload: str
    cycles: int
    toolchain: str
    compiler_flags: tuple[str, ...]
    evidence: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.kernel_id, self.workload, self.toolchain, self.evidence)
        ):
            raise ValueError("kernel cost measurements require complete provenance")
        if isinstance(self.cycles, bool) or not isinstance(self.cycles, int) or self.cycles <= 0:
            raise ValueError("measured kernel cycles must be a positive integer")
        flags = tuple(self.compiler_flags)
        if any(not isinstance(value, str) or not value for value in flags):
            raise ValueError("measured compiler flags must be non-empty strings")
        object.__setattr__(self, "compiler_flags", flags)

    def manifest(self) -> dict[str, object]:
        return {
            "kernel_id": self.kernel_id,
            "workload": self.workload,
            "cycles": self.cycles,
            "toolchain": self.toolchain,
            "compiler_flags": list(self.compiler_flags),
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class TargetDescriptor:
    """Immutable host-side description of a target execution environment.

    The descriptor does not change quantized semantics.  It constrains storage
    alignment, records the intended ISA/toolchain, and carries only explicitly
    measured kernel costs.  Unknown Flash/SRAM capacities remain ``None``
    instead of being guessed from a chip family name.
    """

    target_id: str
    architecture: TargetArchitecture
    cpu: str
    abi: str
    toolchain: str | None
    features: frozenset[str] = field(default_factory=frozenset)
    arena_alignment: int = 1
    constant_alignment: int = 1
    compiler_flags: tuple[str, ...] = ()
    flash_bytes: int | None = None
    sram_bytes: int | None = None
    measured_costs: tuple[KernelCostMeasurement, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.target_id, self.cpu, self.abi)
        ):
            raise ValueError("target id, CPU, and ABI must be non-empty strings")
        if not isinstance(self.architecture, TargetArchitecture):
            raise ValueError("target architecture must use TargetArchitecture")
        if self.toolchain is not None and (
            not isinstance(self.toolchain, str) or not self.toolchain
        ):
            raise ValueError("target toolchain must be None or a non-empty string")
        features = frozenset(self.features)
        if any(not isinstance(value, str) or not value for value in features):
            raise ValueError("target features must be non-empty strings")
        flags = tuple(self.compiler_flags)
        if any(not isinstance(value, str) or not value for value in flags):
            raise ValueError("target compiler flags must be non-empty strings")
        for name, value in (
            ("arena_alignment", self.arena_alignment),
            ("constant_alignment", self.constant_alignment),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                or value & (value - 1)
                or value > _TARGET_SIZE_MAX
            ):
                raise ValueError(f"{name} must be a positive power of two")
        for name, value in (("flash_bytes", self.flash_bytes), ("sram_bytes", self.sram_bytes)):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                or value > _TARGET_SIZE_MAX
            ):
                raise ValueError(f"{name} must be None or a positive integer")
        costs = tuple(self.measured_costs)
        if any(not isinstance(item, KernelCostMeasurement) for item in costs):
            raise ValueError("measured costs must use KernelCostMeasurement")
        cost_keys = {(item.kernel_id, item.workload) for item in costs}
        if len(cost_keys) != len(costs):
            raise ValueError("measured kernel costs must have unique kernel/workload keys")
        metadata = dict(self.metadata)
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in metadata.items()
        ):
            raise ValueError("target metadata requires non-empty string keys and values")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "compiler_flags", flags)
        object.__setattr__(self, "measured_costs", costs)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    @property
    def has_measured_costs(self) -> bool:
        return bool(self.measured_costs)

    def manifest(self) -> dict[str, object]:
        return {
            "id": self.target_id,
            "architecture": self.architecture.value,
            "cpu": self.cpu,
            "abi": self.abi,
            "toolchain": self.toolchain,
            "features": sorted(self.features),
            "arena_alignment": self.arena_alignment,
            "constant_alignment": self.constant_alignment,
            "compiler_flags": list(self.compiler_flags),
            "flash_bytes": self.flash_bytes,
            "sram_bytes": self.sram_bytes,
            "measured_cost_table": {
                "available": self.has_measured_costs,
                "entry_count": len(self.measured_costs),
                "entries": [item.manifest() for item in self.measured_costs],
            },
            "metadata": dict(self.metadata),
        }


__all__ = ["KernelCostMeasurement", "TargetArchitecture", "TargetDescriptor"]
