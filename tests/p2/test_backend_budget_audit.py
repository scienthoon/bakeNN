"""SRAM-aware kernel selection must not reject an otherwise deployable model."""

from dataclasses import replace
import shutil
import subprocess

import numpy as np
import pytest

import bakenn
from bakenn.backend.portable_c import KernelCapability, select_backend_plan
from bakenn.backend.portable_c.selection import canonical_workload_key
from bakenn.errors import CompileError
from bakenn.plan import lower_to_plan
from bakenn.targets import CORTEX_M4, KernelCostMeasurement
from benchmarks.tflm_compare.model_fixtures import cmsis_mlp_graph
from tests.p0.model_fixtures import tiny_cnn_graph
from tests.p0.test_models_c import _compile_runner
from tests.p2.support import require_compiler
from tests.p2.test_backend_selection import linear_graph


def test_static_selection_falls_back_to_sram_feasible_cnn_kernels(tmp_path) -> None:
    graph = tiny_cnn_graph()
    plan = lower_to_plan(graph)
    portable = select_backend_plan(plan, bakenn.CBackendOptions(target=CORTEX_M4))
    target = replace(CORTEX_M4, sram_bytes=portable.arena_size)
    compiled = bakenn.compile(
        graph, tmp_path / "generated", target=target,
        backend_options=bakenn.CBackendOptions(kernel_policy=bakenn.KernelPolicy.STATIC_PRIORITY),
    )
    backend = compiled.artifacts.backend_plan
    assert backend.arena_size <= target.sram_bytes
    assert any("SRAM budget" in reason for entry in backend.selections for reason in entry.rejected.values())
    manifest = bakenn.load_manifest(compiled.artifacts.manifest)
    assert manifest["arena_bytes"] == backend.arena_size
    compiler = require_compiler("clang")
    # The shared guarded runner searches its own directory for the public
    # header. Keep that auxiliary copy outside the manifest-owned closure.
    shutil.copyfile(compiled.artifacts.header, tmp_path / compiled.artifacts.header.name)
    executable = _compile_runner(compiled, tmp_path, compiler)
    sample = np.arange(np.prod(graph.values[graph.inputs[0]].shape), dtype=np.int8).reshape(
        graph.values[graph.inputs[0]].shape
    )
    result = subprocess.run([str(executable)], input=sample.tobytes(), capture_output=True, check=True)
    np.testing.assert_array_equal(np.frombuffer(result.stdout, dtype=np.int8), bakenn.run_reference(compiled.plan, sample).reshape(-1))
    bakenn.load_manifest(compiled.artifacts.manifest)


def _capabilities(monkeypatch, rows):  # type: ignore[no-untyped-def]
    import bakenn.backend.portable_c.selection as selection

    def candidates(step, plan, options):  # type: ignore[no-untyped-def]
        return rows[tuple(item.name for item in plan.steps).index(step.name)]

    monkeypatch.setattr(selection, "kernel_capabilities", candidates)


def _candidate(name, priority, scratch_size=0, scratch_alignment=1, *, optimized=True):  # type: ignore[no-untyped-def]
    return KernelCapability(name, priority, optimized, True, "audit candidate", scratch_size=scratch_size, scratch_alignment=scratch_alignment)


@pytest.mark.parametrize("portable_measured", (False, True))
def test_measured_budget_fallback_preserves_cost_and_rejection_metadata(monkeypatch, portable_measured):
    plan = lower_to_plan(linear_graph())
    fast = _candidate("fast", 30, 65, 8)
    portable = _candidate("portable", 0, optimized=False)
    _capabilities(monkeypatch, ((portable, fast),))
    workload = canonical_workload_key(plan, plan.steps[0])
    fast_cost = KernelCostMeasurement("fast", workload, 10, CORTEX_M4.toolchain, CORTEX_M4.compiler_flags, "fast.json")
    portable_cost = KernelCostMeasurement("portable", workload, 100, CORTEX_M4.toolchain, CORTEX_M4.compiler_flags, "portable.json")
    target = replace(CORTEX_M4, sram_bytes=64, measured_costs=(fast_cost, portable_cost) if portable_measured else (fast_cost,))
    backend = select_backend_plan(plan, bakenn.CBackendOptions(target=target, kernel_policy=bakenn.KernelPolicy.MEASURED))
    selected = backend.selections[0]
    assert backend.arena_size == 0
    assert selected.kernel_id == "portable"
    assert "SRAM budget" in selected.rejected["fast"]
    assert selected.selection_basis == ("measured_latency" if portable_measured else "measured_portable_fallback")
    assert selected.matched_cost == (portable_cost if portable_measured else None)


def test_require_optimized_finds_globally_feasible_scratch_alignment(monkeypatch):
    plan = lower_to_plan(cmsis_mlp_graph((4, 4, 4)))
    _capabilities(monkeypatch, (
        (_candidate("first_preferred", 50, 1, 64), _candidate("first_compact", 40, 1, 8)),
        (_candidate("second_required", 30, 65, 8),),
    ))
    target = replace(CORTEX_M4, sram_bytes=128)
    backend = select_backend_plan(plan, bakenn.CBackendOptions(target=target, kernel_policy=bakenn.KernelPolicy.REQUIRE_OPTIMIZED))
    assert [item.kernel_id for item in backend.selections] == ["first_compact", "second_required"]
    assert backend.arena_size <= 128
    assert all(item.optimized for item in backend.selections)
    assert "SRAM budget" in backend.selections[0].rejected["first_preferred"]


def test_measured_policy_chooses_fastest_feasible_optimized_candidate(monkeypatch):
    plan = lower_to_plan(linear_graph())
    _capabilities(monkeypatch, ((
        _candidate("oversized_fast", 50, 65, 8),
        _candidate("feasible_fast", 10, 32, 8),
        _candidate("feasible_high_priority", 40, 16, 8),
        _candidate("portable", 0, optimized=False),
    ),))
    workload = canonical_workload_key(plan, plan.steps[0])
    costs = tuple(KernelCostMeasurement(
        name, workload, cycles, CORTEX_M4.toolchain, CORTEX_M4.compiler_flags, f"{name}.json",
    ) for name, cycles in (
        ("oversized_fast", 1), ("feasible_fast", 10), ("feasible_high_priority", 20), ("portable", 100),
    ))
    target = replace(CORTEX_M4, sram_bytes=64, measured_costs=costs)
    backend = select_backend_plan(plan, bakenn.CBackendOptions(target=target, kernel_policy=bakenn.KernelPolicy.MEASURED))
    selected = backend.selections[0]
    assert selected.kernel_id == "feasible_fast"
    assert selected.matched_cost == costs[1]
    assert selected.selection_basis == "measured_latency"
    assert "SRAM budget" in selected.rejected["oversized_fast"]
    assert "20 cycles did not beat" in selected.rejected["feasible_high_priority"]


def test_require_optimized_does_not_silently_use_portable_under_budget(monkeypatch):
    plan = lower_to_plan(linear_graph())
    _capabilities(monkeypatch, ((_candidate("portable", 0, optimized=False), _candidate("too_large", 30, 65, 8)),))
    target = replace(CORTEX_M4, sram_bytes=64)
    with pytest.raises(CompileError, match="require_optimized.*SRAM budget"):
        select_backend_plan(plan, bakenn.CBackendOptions(target=target, kernel_policy=bakenn.KernelPolicy.REQUIRE_OPTIMIZED))


def test_no_feasible_activation_arena_fails_before_creating_artifacts(tmp_path):
    graph = tiny_cnn_graph()
    target = replace(CORTEX_M4, sram_bytes=1)
    output = tmp_path / "too_small"
    with pytest.raises(CompileError, match="SRAM budget"):
        bakenn.compile(graph, output, target=target, backend_options=bakenn.CBackendOptions(kernel_policy=bakenn.KernelPolicy.STATIC_PRIORITY))
    assert not output.exists()
