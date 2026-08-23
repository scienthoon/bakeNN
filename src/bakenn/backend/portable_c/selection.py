from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from functools import singledispatch
import json
from types import MappingProxyType
from typing import Mapping
import warnings

import numpy as np

from bakenn.errors import CompileError
from bakenn.ir.types import PerAxisQParams, PerTensorQParams, TARGET_SIZE_MAX
from bakenn.plan import ExecutionPlan, ExecutionStep
from bakenn.targets import KernelCostMeasurement, PORTABLE_32, TargetDescriptor


class KernelPolicy(str, Enum):
    """How the C backend chooses among semantically equivalent kernels."""

    PORTABLE = "portable"
    STATIC_PRIORITY = "static_priority"
    MEASURED = "measured"
    # Deprecated compatibility spelling for the pre-1.0 static-priority mode.
    AUTO = "auto"
    REQUIRE_OPTIMIZED = "require_optimized"


@dataclass(frozen=True)
class CBackendOptions:
    """Host-side C lowering policy.

    Portable remains the default until a target-specific benchmark validates a
    specialized implementation. ``STATIC_PRIORITY`` is the explicit opt-in to
    deterministic capability-priority selection.  ``AUTO`` retains that same
    behavior only as a deprecated compatibility spelling.  ``MEASURED`` uses
    exact physical-cost entries and otherwise falls back to portable C.
    """

    kernel_policy: KernelPolicy = KernelPolicy.PORTABLE
    enable_weight_packing: bool = True
    enable_cmsis_nn: bool = False
    enable_esp_nn: bool = False
    target: TargetDescriptor = PORTABLE_32

    def __post_init__(self) -> None:
        if not isinstance(self.kernel_policy, KernelPolicy):
            raise ValueError("kernel_policy must use KernelPolicy")
        if self.kernel_policy is KernelPolicy.AUTO:
            warnings.warn(
                "KernelPolicy.AUTO is deprecated because it means static priority, "
                "not measured fastest; use KernelPolicy.STATIC_PRIORITY or "
                "KernelPolicy.MEASURED",
                DeprecationWarning,
                stacklevel=2,
            )
        if not isinstance(self.enable_weight_packing, bool):
            raise ValueError("enable_weight_packing must be boolean")
        if not isinstance(self.enable_cmsis_nn, bool):
            raise ValueError("enable_cmsis_nn must be boolean")
        if not isinstance(self.enable_esp_nn, bool):
            raise ValueError("enable_esp_nn must be boolean")
        if not isinstance(self.target, TargetDescriptor):
            raise ValueError("target must be a TargetDescriptor")


@dataclass(frozen=True, eq=False)
class PackedConstant:
    """A backend-owned immutable representation of one semantic constant."""

    name: str
    source: str
    layout: str
    value: np.ndarray
    alignment: int = 1

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.name, self.source, self.layout)
        ):
            raise ValueError("packed constants require name, source, and layout")
        array = np.array(self.value, copy=True, order="C")
        if array.dtype not in (np.dtype(np.int8), np.dtype(np.int32)):
            raise ValueError("packed constants must use int8 or int32 storage")
        if array.nbytes > TARGET_SIZE_MAX:
            raise ValueError(
                "packed constant storage exceeds the 32-bit target byte limit"
            )
        if (
            isinstance(self.alignment, bool)
            or not isinstance(self.alignment, int)
            or self.alignment <= 0
            or self.alignment & (self.alignment - 1)
            or self.alignment > TARGET_SIZE_MAX
        ):
            raise ValueError("packed constant alignment must be a positive power of two")
        array.setflags(write=False)
        object.__setattr__(self, "value", array)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PackedConstant):
            return NotImplemented
        return (
            self.name == other.name
            and self.source == other.source
            and self.layout == other.layout
            and self.alignment == other.alignment
            and self.value.dtype == other.value.dtype
            and self.value.shape == other.value.shape
            and self.value.tobytes() == other.value.tobytes()
        )


@dataclass(frozen=True)
class KernelCapability:
    """One candidate implementation and its exact applicability result."""

    kernel_id: str
    priority: int
    optimized: bool
    supported: bool
    reason: str
    packed_constants: tuple[PackedConstant, ...] = ()
    constant_overrides: Mapping[str, str] = field(default_factory=dict)
    scratch_size: int = 0
    scratch_alignment: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.kernel_id, str)
            or not self.kernel_id
            or not isinstance(self.reason, str)
            or not self.reason
        ):
            raise ValueError("kernel capabilities require an id and reason")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ValueError("kernel priority must be an integer")
        if not isinstance(self.optimized, bool) or not isinstance(self.supported, bool):
            raise ValueError("kernel capability flags must be boolean")
        if (
            isinstance(self.scratch_size, bool)
            or not isinstance(self.scratch_size, int)
            or self.scratch_size < 0
            or self.scratch_size > TARGET_SIZE_MAX
        ):
            raise ValueError(
                "kernel scratch size must fit the 32-bit target byte range"
            )
        if (
            isinstance(self.scratch_alignment, bool)
            or not isinstance(self.scratch_alignment, int)
            or self.scratch_alignment <= 0
            or self.scratch_alignment & (self.scratch_alignment - 1)
            or self.scratch_alignment > TARGET_SIZE_MAX
        ):
            raise ValueError("kernel scratch alignment must be a positive power of two")
        if not self.scratch_size and self.scratch_alignment != 1:
            raise ValueError("zero-sized kernel scratch must use alignment one")
        packed = tuple(self.packed_constants)
        names = {item.name for item in packed}
        if len(names) != len(packed):
            raise ValueError("packed constant names must be unique within a candidate")
        overrides = dict(self.constant_overrides)
        if any(
            not isinstance(source, str)
            or not source
            or not isinstance(target, str)
            or not target
            for source, target in overrides.items()
        ):
            raise ValueError("constant overrides require non-empty string names")
        if set(overrides.values()) - names:
            raise ValueError("constant overrides must reference declared packed constants")
        packed_by_name = {item.name: item for item in packed}
        for source, target in overrides.items():
            if packed_by_name[target].source != source:
                raise ValueError(
                    "constant override source must match its packed representation source"
                )
        if not self.supported and (packed or overrides):
            raise ValueError("unsupported candidates cannot carry packed representations")
        object.__setattr__(self, "packed_constants", packed)
        object.__setattr__(self, "constant_overrides", MappingProxyType(overrides))


_WORKLOAD_KEY_VERSION = "bakenn.workload.v1"
_WORKLOAD_PARAMETER_FIELDS = frozenset(
    {
        "activation_max",
        "activation_min",
        "align_corners",
        "axis",
        "axis_sizes",
        "channels",
        "class_count",
        "depth_multiplier",
        "dilation",
        "groups",
        "inner_size",
        "inplace",
        "input_axis_size",
        "kernel",
        "materialize",
        "operation",
        "outer_size",
        "output_axis_size",
        "output_padding",
        "padding",
        "position_count",
        "row_count",
        "start",
        "step",
        "stride",
    }
)


def _canonical_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_canonical_value(item) for item in value]
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise TypeError(f"unsupported canonical workload value {type(value).__name__}")


def _zero_point_profile(values: tuple[int, ...]) -> str:
    if all(value == 0 for value in values):
        return "symmetric_zero"
    if all(value == values[0] for value in values):
        return "uniform_nonzero"
    return "per_axis_nonzero"


def _tensor_workload_descriptor(plan: ExecutionPlan, name: str) -> dict[str, object]:
    tensor = plan.tensors[name]
    tensor_type = tensor.tensor_type
    qparams = tensor_type.qparams
    if isinstance(qparams, PerTensorQParams):
        qparam_profile: dict[str, object] = {
            "granularity": "per_tensor",
            "zero_point_profile": (
                "symmetric_zero" if qparams.zero_point == 0 else "asymmetric_nonzero"
            ),
        }
    elif isinstance(qparams, PerAxisQParams):
        qparam_profile = {
            "granularity": "per_axis",
            "axis": qparams.axis,
            "count": len(qparams.scales),
            "zero_point_profile": _zero_point_profile(qparams.zero_points),
        }
    else:  # pragma: no cover - TensorType validation keeps this fail-closed.
        raise TypeError(f"unsupported qparams type {type(qparams).__name__}")
    return {
        "shape": list(tensor_type.shape),
        "dtype": tensor_type.dtype.value,
        "layout": tensor_type.layout.value,
        "qparams": qparam_profile,
        "memory": {
            "storage": tensor.storage.value,
            "bytes": tensor_type.nbytes,
        },
    }


def _requantization_profile(step: ExecutionStep) -> dict[str, object]:
    shift_values: list[int] = []
    multiplier_count = 0
    for item in fields(step):
        value = getattr(step, item.name)
        if item.name == "multipliers":
            multiplier_count += len(value)
        elif item.name == "multiplier" or item.name.endswith("_multiplier"):
            multiplier_count += 1
        if item.name == "shifts":
            shift_values.extend(int(shift) for shift in value)
        elif item.name == "shift" or item.name.endswith("_shift"):
            shift_values.append(int(value))
    return {
        "multiplier_count": multiplier_count,
        "shift_count": len(shift_values),
        "negative_shifts": sum(value < 0 for value in shift_values),
        "zero_shifts": sum(value == 0 for value in shift_values),
        "positive_shifts": sum(value > 0 for value in shift_values),
    }


def canonical_workload_key(plan: ExecutionPlan, step: ExecutionStep) -> str:
    """Return the versioned exact key used by physical kernel-cost entries.

    Model and tensor names are deliberately excluded.  The key captures the op
    kind, arithmetic/qparam profile, relevant static parameters, tensor shapes
    and layouts, and physical storage classes.  Canonical JSON keeps the entry
    both deterministic and reviewable in target manifests.
    """

    if not isinstance(plan, ExecutionPlan):
        raise TypeError("canonical workload keys require an ExecutionPlan")
    if not isinstance(step, ExecutionStep):
        raise TypeError("canonical workload keys require an ExecutionStep")
    parameters = {
        item.name: _canonical_value(getattr(step, item.name))
        for item in fields(step)
        if item.name in _WORKLOAD_PARAMETER_FIELDS
    }
    payload = {
        "op": step.kernel_kind,
        "arithmetic_profile": step.arithmetic_profile,
        "inputs": [
            _tensor_workload_descriptor(plan, name) for name in step.inputs
        ],
        "outputs": [
            _tensor_workload_descriptor(plan, name) for name in step.outputs
        ],
        "constants": [
            _tensor_workload_descriptor(plan, name) for name in step.constants
        ],
        "parameters": parameters,
        "requantization": _requantization_profile(step),
        "memory": {
            "step_scratch_bytes": step.scratch_size,
            "step_scratch_alignment": step.scratch_alignment,
        },
    }
    return _WORKLOAD_KEY_VERSION + ":" + json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


@dataclass(frozen=True)
class KernelSelection:
    step_index: int
    step_name: str
    kernel_id: str
    optimized: bool
    reason: str
    rejected: Mapping[str, str] = field(default_factory=dict)
    constant_overrides: Mapping[str, str] = field(default_factory=dict)
    packed_constants: tuple[PackedConstant, ...] = ()
    scratch_size: int = 0
    scratch_alignment: int = 1
    selection_basis: str = "manual"
    workload_key: str = ""
    matched_cost: KernelCostMeasurement | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or self.step_index < 0
            or not isinstance(self.step_name, str)
            or not self.step_name
            or not isinstance(self.kernel_id, str)
            or not self.kernel_id
            or not isinstance(self.reason, str)
            or not self.reason
        ):
            raise ValueError("kernel selection is incomplete")
        if not isinstance(self.optimized, bool):
            raise ValueError("kernel selection optimized flag must be boolean")
        if not isinstance(self.selection_basis, str) or not self.selection_basis:
            raise ValueError("kernel selection basis must be a non-empty string")
        if not isinstance(self.workload_key, str):
            raise ValueError("kernel selection workload key must be a string")
        if self.workload_key and not self.workload_key.startswith(
            _WORKLOAD_KEY_VERSION + ":"
        ):
            raise ValueError("kernel selection workload key uses an unknown version")
        if self.matched_cost is not None:
            if not isinstance(self.matched_cost, KernelCostMeasurement):
                raise ValueError(
                    "kernel selection matched cost must use KernelCostMeasurement"
                )
            if self.matched_cost.kernel_id != self.kernel_id:
                raise ValueError("matched cost kernel id must match the selected kernel")
            if self.matched_cost.workload != self.workload_key:
                raise ValueError("matched cost workload must match the selection key")
            if not self.selection_basis.startswith("measured"):
                raise ValueError("matched costs require a measured selection basis")
        if (
            isinstance(self.scratch_size, bool)
            or not isinstance(self.scratch_size, int)
            or self.scratch_size < 0
            or self.scratch_size > TARGET_SIZE_MAX
        ):
            raise ValueError(
                "selected kernel scratch size must fit the 32-bit target byte range"
            )
        if (
            isinstance(self.scratch_alignment, bool)
            or not isinstance(self.scratch_alignment, int)
            or self.scratch_alignment <= 0
            or self.scratch_alignment & (self.scratch_alignment - 1)
            or self.scratch_alignment > TARGET_SIZE_MAX
        ):
            raise ValueError("selected kernel scratch alignment must be a power of two")
        if not self.scratch_size and self.scratch_alignment != 1:
            raise ValueError("zero-sized selected scratch must use alignment one")
        packed = tuple(self.packed_constants)
        packed_by_name = {item.name: item for item in packed}
        if len(packed_by_name) != len(packed):
            raise ValueError("selected packed constant names must be unique")
        overrides = dict(self.constant_overrides)
        if set(overrides.values()) - set(packed_by_name):
            raise ValueError("selected overrides must reference selected packed constants")
        for source, target in overrides.items():
            if packed_by_name[target].source != source:
                raise ValueError(
                    "selected override source must match its packed representation source"
                )
        object.__setattr__(self, "rejected", MappingProxyType(dict(self.rejected)))
        object.__setattr__(
            self, "constant_overrides", MappingProxyType(overrides)
        )
        object.__setattr__(self, "packed_constants", packed)


@dataclass(frozen=True)
class CBackendPlan:
    """Backend decisions layered on top of an unchanged semantic plan."""

    execution_plan: ExecutionPlan
    options: CBackendOptions
    selections: tuple[KernelSelection, ...]
    packed_constants: Mapping[str, PackedConstant]
    activation_arena_size: int = field(init=False)
    scratch_size: int = field(init=False)
    scratch_offset: int | None = field(init=False)
    scratch_alignment: int = field(init=False)
    arena_size: int = field(init=False)
    arena_alignment: int = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.execution_plan, ExecutionPlan):
            raise TypeError("backend plan requires an ExecutionPlan")
        if not isinstance(self.options, CBackendOptions):
            raise TypeError("backend plan requires CBackendOptions")
        selections = tuple(self.selections)
        if len(selections) != len(self.execution_plan.steps):
            raise CompileError("backend plan must select exactly one kernel per execution step")
        if tuple(item.step_index for item in selections) != tuple(range(len(selections))):
            raise CompileError("backend selections must follow execution order")
        for selection, step in zip(selections, self.execution_plan.steps):
            if selection.step_name != step.name:
                raise CompileError("backend selection step names must match execution steps")
        packed = dict(self.packed_constants)
        if any(name != item.name for name, item in packed.items()):
            raise CompileError("packed constant mapping keys must match their names")
        all_declared: set[str] = set()
        for selection, step in zip(selections, self.execution_plan.steps):
            declared_by_name = {
                item.name: item for item in selection.packed_constants
            }
            declared = set(declared_by_name)
            all_declared.update(declared)
            unknown_sources = {
                item.source for item in selection.packed_constants
            } - set(step.constants)
            if unknown_sources:
                raise CompileError(
                    f"backend selection packs non-step constants {sorted(unknown_sources)}"
                )
            unknown_overrides = set(selection.constant_overrides) - set(step.constants)
            if unknown_overrides:
                raise CompileError(
                    f"backend selection overrides non-step constants "
                    f"{sorted(unknown_overrides)}"
                )
            if set(selection.constant_overrides.values()) - declared:
                raise CompileError("backend selection override lacks its packed constant")
            for source, target in selection.constant_overrides.items():
                if declared_by_name[target].source != source:
                    raise CompileError(
                        "backend selection override source does not match packed source"
                    )
            if declared - set(packed):
                raise CompileError("backend selection packed constant is absent from backend plan")
            if any(packed[item.name] != item for item in selection.packed_constants):
                raise CompileError("backend selection packed constant conflicts with backend plan")
        if set(packed) != all_declared:
            raise CompileError(
                "backend packed constant mapping must exactly match selected representations"
            )
        activation_size = self.execution_plan.activation_arena_size
        scratch_size = max(
            (self.execution_plan.scratch_size, *(item.scratch_size for item in selections))
        )
        scratch_alignment = max(
            (
                self.execution_plan.scratch_alignment,
                *(item.scratch_alignment for item in selections if item.scratch_size),
            )
        )
        arena_alignment = max(
            self.execution_plan.arena_alignment,
            scratch_alignment,
            self.options.target.arena_alignment,
        )
        if arena_alignment > TARGET_SIZE_MAX:
            raise CompileError("backend arena alignment exceeds the 32-bit target limit")
        if scratch_size:
            scratch_offset = (
                activation_size + scratch_alignment - 1
            ) & -scratch_alignment
            arena_end = scratch_offset + scratch_size
            arena_size = (arena_end + arena_alignment - 1) & -arena_alignment
            if (
                scratch_offset > TARGET_SIZE_MAX
                or arena_end > TARGET_SIZE_MAX
                or arena_size > TARGET_SIZE_MAX
            ):
                raise CompileError(
                    "backend scratch or arena exceeds the 32-bit target byte limit"
                )
        else:
            scratch_offset = None
            arena_size = (
                self.execution_plan.arena_size + arena_alignment - 1
            ) & -arena_alignment
            if arena_size > TARGET_SIZE_MAX:
                raise CompileError(
                    "target arena alignment makes the arena exceed the 32-bit byte limit"
                )
        object.__setattr__(self, "selections", selections)
        object.__setattr__(self, "packed_constants", MappingProxyType(packed))
        object.__setattr__(self, "activation_arena_size", activation_size)
        object.__setattr__(self, "scratch_size", scratch_size)
        object.__setattr__(self, "scratch_offset", scratch_offset)
        object.__setattr__(self, "scratch_alignment", scratch_alignment)
        object.__setattr__(self, "arena_size", arena_size)
        object.__setattr__(self, "arena_alignment", arena_alignment)


def _portable_capability(step: ExecutionStep) -> KernelCapability:
    return KernelCapability(
        kernel_id=f"portable.{step.kernel_kind}.v1",
        priority=0,
        optimized=False,
        supported=True,
        reason="portable C baseline is defined for this lowered step",
    )


@singledispatch
def kernel_capabilities(
    step: object,
    plan: ExecutionPlan,
    options: CBackendOptions,
) -> tuple[KernelCapability, ...]:
    del plan, options
    if not isinstance(step, ExecutionStep):
        raise CompileError(f"cannot select a C kernel for {type(step).__name__}")
    return (_portable_capability(step),)


@dataclass(frozen=True)
class _KernelDecision:
    capability: KernelCapability
    selection_basis: str
    reason: str
    matched_cost: KernelCostMeasurement | None = None


def _exact_measured_costs(
    capabilities: tuple[KernelCapability, ...],
    workload_key: str,
    options: CBackendOptions,
) -> dict[str, KernelCostMeasurement]:
    """Return only cost entries matching kernel, workload, toolchain and flags."""

    target = options.target
    if target.toolchain is None:
        return {}
    supported_ids = {
        capability.kernel_id for capability in capabilities if capability.supported
    }
    return {
        measurement.kernel_id: measurement
        for measurement in target.measured_costs
        if measurement.kernel_id in supported_ids
        and measurement.workload == workload_key
        and measurement.toolchain == target.toolchain
        and measurement.compiler_flags == target.compiler_flags
    }


def _choose(
    step: ExecutionStep,
    capabilities: tuple[KernelCapability, ...],
    options: CBackendOptions,
    workload_key: str,
) -> _KernelDecision:
    if not capabilities:
        raise CompileError(f"{step.name}: no C kernel candidates were registered")
    identifiers = [item.kernel_id for item in capabilities]
    if len(set(identifiers)) != len(identifiers):
        raise CompileError(f"{step.name}: duplicate C kernel candidate identifiers")
    supported = [item for item in capabilities if item.supported]
    if options.kernel_policy is KernelPolicy.PORTABLE:
        supported = [item for item in supported if not item.optimized]
    elif options.kernel_policy is KernelPolicy.REQUIRE_OPTIMIZED:
        supported = [item for item in supported if item.optimized]
    elif options.kernel_policy is KernelPolicy.MEASURED:
        exact_costs = _exact_measured_costs(capabilities, workload_key, options)
        measured = [
            (item, exact_costs[item.kernel_id])
            for item in supported
            if item.kernel_id in exact_costs
        ]
        if measured:
            chosen, cost = sorted(
                measured,
                key=lambda item: (
                    item[1].cycles,
                    item[0].optimized,
                    -item[0].priority,
                    item[0].kernel_id,
                ),
            )[0]
            flags = ",".join(cost.compiler_flags)
            return _KernelDecision(
                capability=chosen,
                selection_basis="measured_latency",
                reason=(
                    f"{chosen.reason}; selected by exact measured latency "
                    f"({cost.cycles} cycles, toolchain={cost.toolchain}, "
                    f"flags=[{flags}], evidence={cost.evidence}, "
                    f"workload={workload_key})"
                ),
                matched_cost=cost,
            )
        supported = [item for item in supported if not item.optimized]
    if not supported:
        rejected = "; ".join(
            f"{item.kernel_id}: {item.reason}" for item in capabilities if not item.supported
        )
        policy = options.kernel_policy.value
        raise CompileError(
            f"{step.name}: kernel policy {policy} has no supported implementation"
            + (f" ({rejected})" if rejected else "")
        )
    chosen = sorted(supported, key=lambda item: (-item.priority, item.kernel_id))[0]
    if options.kernel_policy is KernelPolicy.MEASURED:
        return _KernelDecision(
            capability=chosen,
            selection_basis="measured_portable_fallback",
            reason=(
                f"{chosen.reason}; no exact measured latency cost matched the "
                f"canonical workload, target toolchain and compiler flags, so the "
                f"selector used portable C (workload={workload_key})"
            ),
        )
    if options.kernel_policy is KernelPolicy.PORTABLE:
        basis = "portable_policy"
    elif options.kernel_policy is KernelPolicy.REQUIRE_OPTIMIZED:
        basis = "require_optimized"
    else:
        basis = "static_priority"
    return _KernelDecision(
        capability=chosen,
        selection_basis=basis,
        reason=chosen.reason,
    )


def select_backend_plan(
    plan: ExecutionPlan,
    options: CBackendOptions | None = None,
) -> CBackendPlan:
    """Select kernels and representations deterministically, without mutating IR."""

    # Importing the aggregator installs family-specific capability functions.
    from . import families as _families  # noqa: F401

    resolved_options = CBackendOptions() if options is None else options
    if not isinstance(resolved_options, CBackendOptions):
        raise TypeError("options must be CBackendOptions")
    selections: list[KernelSelection] = []
    packed: dict[str, PackedConstant] = {}
    for index, step in enumerate(plan.steps):
        capabilities = tuple(kernel_capabilities(step, plan, resolved_options))
        workload_key = canonical_workload_key(plan, step)
        decision = _choose(step, capabilities, resolved_options, workload_key)
        chosen = decision.capability
        measured_costs = _exact_measured_costs(
            capabilities, workload_key, resolved_options
        )
        rejected: dict[str, str] = {}
        for item in capabilities:
            if item.kernel_id == chosen.kernel_id:
                continue
            if not item.supported:
                rejected[item.kernel_id] = item.reason
            elif resolved_options.kernel_policy is KernelPolicy.PORTABLE and item.optimized:
                rejected[item.kernel_id] = "excluded by portable kernel policy"
            elif (
                resolved_options.kernel_policy is KernelPolicy.REQUIRE_OPTIMIZED
                and not item.optimized
            ):
                rejected[item.kernel_id] = "excluded by require_optimized kernel policy"
            elif resolved_options.kernel_policy is KernelPolicy.MEASURED:
                measured_cost = measured_costs.get(item.kernel_id)
                if measured_cost is None:
                    rejected[item.kernel_id] = (
                        "no exact measured latency cost matches the canonical workload, "
                        "target toolchain and compiler flags"
                    )
                elif decision.matched_cost is not None:
                    rejected[item.kernel_id] = (
                        f"measured latency {measured_cost.cycles} cycles did not beat "
                        f"{chosen.kernel_id} at {decision.matched_cost.cycles} cycles"
                    )
                else:  # pragma: no cover - a match always produces a measured decision.
                    rejected[item.kernel_id] = "not selected by measured latency policy"
            else:
                rejected[item.kernel_id] = (
                    f"lower selection priority than {chosen.kernel_id}"
                )
        for item in chosen.packed_constants:
            if item.name in plan.constants:
                raise CompileError(f"packed constant {item.name} collides with semantic storage")
            if item.source not in plan.constants:
                raise CompileError(
                    f"{step.name}: packed constant {item.name} has unknown source {item.source}"
                )
            if item.source not in step.constants:
                raise CompileError(
                    f"{step.name}: packed constant {item.name} derives from "
                    f"non-step constant {item.source}"
                )
            previous = packed.setdefault(item.name, item)
            if (
                previous.source != item.source
                or previous.layout != item.layout
                or previous.alignment != item.alignment
                or previous.value.dtype != item.value.dtype
                or previous.value.shape != item.value.shape
                or previous.value.tobytes() != item.value.tobytes()
            ):
                raise CompileError(f"conflicting packed constant {item.name}")
        unknown_overrides = set(chosen.constant_overrides) - set(step.constants)
        if unknown_overrides:
            raise CompileError(
                f"{step.name}: kernel overrides non-step constants "
                f"{sorted(unknown_overrides)}"
            )
        selections.append(
            KernelSelection(
                step_index=index,
                step_name=step.name,
                kernel_id=chosen.kernel_id,
                optimized=chosen.optimized,
                reason=decision.reason,
                rejected=rejected,
                constant_overrides=chosen.constant_overrides,
                packed_constants=chosen.packed_constants,
                scratch_size=chosen.scratch_size,
                scratch_alignment=chosen.scratch_alignment,
                selection_basis=decision.selection_basis,
                workload_key=workload_key,
                matched_cost=decision.matched_cost,
            )
        )
    return CBackendPlan(plan, resolved_options, tuple(selections), packed)


__all__ = [
    "CBackendOptions",
    "CBackendPlan",
    "KernelCapability",
    "KernelPolicy",
    "KernelSelection",
    "PackedConstant",
    "canonical_workload_key",
    "kernel_capabilities",
    "select_backend_plan",
]
