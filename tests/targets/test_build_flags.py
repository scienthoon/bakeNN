from dataclasses import replace
import shutil

import pytest

import bakenn
from bakenn.backend.portable_c.selection import canonical_workload_key
from bakenn.errors import CompileError
from bakenn.plan import lower_to_plan
from bakenn.targets import CORTEX_M4, KernelCostMeasurement
from tests.p2.test_backend_selection import linear_graph


def _measured_model(tmp_path, optimization="-O3"):
    graph = linear_graph()
    plan = lower_to_plan(graph)
    target = replace(
        CORTEX_M4,
        compiler_flags=(*CORTEX_M4.compiler_flags, *((optimization,) if optimization else ())),
    )
    measurement = KernelCostMeasurement(
        "optimized.linear_oi2.v1", canonical_workload_key(plan, plan.steps[0]),
        123, target.toolchain, target.compiler_flags, "physical-O3-result.json",
    )
    target = replace(target, measured_costs=(measurement,))
    compiled = bakenn.compile(
        graph, tmp_path / "generated",
        backend_options=bakenn.CBackendOptions(
            kernel_policy=bakenn.KernelPolicy.MEASURED, target=target,
        ),
    )
    assert compiled.artifacts.backend_plan.selections[0].selection_basis == "measured_latency"
    return compiled, target


@pytest.mark.parametrize("optimization", ("-O3", None))
def test_build_preserves_declared_optimization_by_default(tmp_path, optimization):
    if shutil.which("arm-none-eabi-gcc") is None:
        pytest.skip("ARM cross compiler is not installed")
    compiled, target = _measured_model(tmp_path, optimization)
    report = bakenn.build_freestanding_elf(compiled.artifacts, target, tmp_path / "build")
    assert [flag for flag in report.compiler_flags if flag.startswith("-O")] == (
        [optimization] if optimization else []
    )
    assert report.elf.is_file()


@pytest.mark.parametrize("change", ("optimization", "extra_flags", "target"))
def test_measured_build_rejects_changed_flags_before_writing(tmp_path, change):
    compiled, target = _measured_model(tmp_path)
    kwargs = {}
    if change == "optimization":
        kwargs["optimization"] = "-O2"
    elif change == "extra_flags":
        kwargs["extra_flags"] = ("-O0",)
    else:
        target = replace(target, compiler_flags=(*CORTEX_M4.compiler_flags, "-Os"))
    output = tmp_path / "build"
    with pytest.raises(CompileError, match="measured.*flags|flags.*measured"):
        bakenn.build_freestanding_elf(compiled.artifacts, target, output, **kwargs)
    assert not output.exists()
