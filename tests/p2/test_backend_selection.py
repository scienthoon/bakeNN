from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

import bakenn
from bakenn.backend.portable_c import (
    CBackendPlan,
    KernelCapability,
    KernelSelection,
    PackedConstant,
    StepEmitContext,
    select_backend_plan,
)
from bakenn.backend.portable_c.selection import canonical_workload_key
from bakenn.errors import CompileError
from bakenn.ir import (
    DType,
    Layout,
    LinearOp,
    PerAxisQParams,
    PerTensorQParams,
    QuantizedGraph,
    TensorType,
)
from bakenn.plan import lower_to_plan
from bakenn.ir.types import TARGET_SIZE_MAX
from bakenn.targets import CORTEX_M4, KernelCostMeasurement


def linear_graph(input_count: int = 12, output_count: int = 6) -> QuantizedGraph:
    input_q = PerTensorQParams(0.5, -7)
    output_q = PerTensorQParams(0.01, 3)
    weight_scales = tuple(0.05 + channel * 0.01 for channel in range(output_count))
    weight = (
        (np.arange(input_count * output_count, dtype=np.int16) * 7 + 3) % 23 - 11
    ).reshape(output_count, input_count).astype(np.int8)
    bias = (np.arange(output_count, dtype=np.int32) * 37 - 51).astype(np.int32)
    return QuantizedGraph(
        name="p2_linear",
        values={
            "input": TensorType((1, input_count), DType.INT8, Layout.NC, input_q),
            "weight": TensorType(
                (output_count, input_count),
                DType.INT8,
                Layout.OI,
                PerAxisQParams(weight_scales, (0,) * output_count, 0),
            ),
            "bias": TensorType(
                (output_count,),
                DType.INT32,
                Layout.C,
                PerAxisQParams(
                    tuple(input_q.scale * scale for scale in weight_scales),
                    (0,) * output_count,
                    0,
                ),
            ),
            "output": TensorType((1, output_count), DType.INT8, Layout.NC, output_q),
        },
        constants={"weight": weight, "bias": bias},
        ops=(LinearOp("linear", "input", "weight", "bias", "output"),),
        inputs=("input",),
        outputs=("output",),
    )


def test_portable_is_default_and_static_priority_selects_deterministic_packed_linear() -> None:
    plan = lower_to_plan(linear_graph())
    portable = select_backend_plan(plan)
    assert portable.selections[0].kernel_id == "portable.linear_s8.v1"
    assert not portable.selections[0].optimized
    assert not portable.packed_constants

    options = bakenn.CBackendOptions(
        kernel_policy=bakenn.KernelPolicy.STATIC_PRIORITY
    )
    first = select_backend_plan(plan, options)
    second = select_backend_plan(plan, options)
    selection = first.selections[0]
    assert selection.kernel_id == "optimized.linear_oi2.v1"
    assert selection.optimized
    assert selection.constant_overrides == {
        "weight": "weight.linear_oi2",
    }
    packed = first.packed_constants["weight.linear_oi2"]
    expected = np.ascontiguousarray(plan.constants["weight"].reshape(3, 2, 12).transpose(0, 2, 1))
    np.testing.assert_array_equal(packed.value, expected)
    assert packed.layout == "linear_oi2_interleaved_v1"
    assert not packed.value.flags.writeable
    assert first.selections == second.selections
    assert first.execution_plan is plan
    np.testing.assert_array_equal(plan.constants["weight"], linear_graph().constants["weight"])


def test_auto_is_a_deprecated_compatibility_spelling_for_static_priority() -> None:
    plan = lower_to_plan(linear_graph())
    with pytest.warns(DeprecationWarning, match="STATIC_PRIORITY"):
        auto_options = bakenn.CBackendOptions(
            kernel_policy=bakenn.KernelPolicy.AUTO
        )
    auto = select_backend_plan(plan, auto_options)
    explicit = select_backend_plan(
        plan,
        bakenn.CBackendOptions(
            kernel_policy=bakenn.KernelPolicy.STATIC_PRIORITY
        ),
    )
    assert auto.selections[0].kernel_id == explicit.selections[0].kernel_id
    assert auto.selections[0].selection_basis == "static_priority"
    assert auto.selections[0].matched_cost is None


def test_canonical_workload_key_is_versioned_and_independent_of_names() -> None:
    plan = lower_to_plan(linear_graph())
    key = canonical_workload_key(plan, plan.steps[0])
    renamed_step = replace(plan.steps[0], name="renamed_linear")
    renamed_plan = replace(plan, name="renamed_model", steps=(renamed_step,))
    assert canonical_workload_key(renamed_plan, renamed_step) == key
    assert key.startswith("bakenn.workload.v1:")
    assert '"op":"linear_s8"' in key
    assert '"shape":[1,12]' in key
    assert '"storage":"input"' in key
    different = lower_to_plan(linear_graph(input_count=13))
    assert canonical_workload_key(different, different.steps[0]) != key


def _measured_cost(
    kernel_id: str,
    workload: str,
    cycles: int,
    evidence: str,
    *,
    toolchain: str | None = CORTEX_M4.toolchain,
    compiler_flags: tuple[str, ...] = CORTEX_M4.compiler_flags,
) -> KernelCostMeasurement:
    assert toolchain is not None
    return KernelCostMeasurement(
        kernel_id=kernel_id,
        workload=workload,
        cycles=cycles,
        toolchain=toolchain,
        compiler_flags=compiler_flags,
        evidence=evidence,
    )


def test_measured_policy_uses_only_exact_costs_and_records_provenance() -> None:
    plan = lower_to_plan(linear_graph())
    workload = canonical_workload_key(plan, plan.steps[0])
    portable_cost = _measured_cost(
        "portable.linear_s8.v1", workload, 400, "portable-run.json"
    )
    generic_cost = _measured_cost(
        "optimized.linear_oi2.v1", workload, 100, "oi2-run.json"
    )
    m4_cost = _measured_cost(
        "cortex_m4.linear_smlad.v1", workload, 150, "smlad-run.json"
    )
    target = replace(
        CORTEX_M4,
        measured_costs=(portable_cost, generic_cost, m4_cost),
    )
    backend = select_backend_plan(
        plan,
        bakenn.CBackendOptions(
            kernel_policy=bakenn.KernelPolicy.MEASURED,
            target=target,
        ),
    )
    selection = backend.selections[0]
    assert selection.kernel_id == "optimized.linear_oi2.v1"
    assert selection.selection_basis == "measured_latency"
    assert selection.workload_key == workload
    assert selection.matched_cost == generic_cost
    assert "100 cycles" in selection.reason
    assert "oi2-run.json" in selection.reason
    assert "150 cycles did not beat" in selection.rejected[
        "cortex_m4.linear_smlad.v1"
    ]


def test_measured_policy_falls_back_for_empty_or_stale_cost_table() -> None:
    plan = lower_to_plan(linear_graph())
    workload = canonical_workload_key(plan, plan.steps[0])
    empty = select_backend_plan(
        plan,
        bakenn.CBackendOptions(
            kernel_policy=bakenn.KernelPolicy.MEASURED,
            target=CORTEX_M4,
        ),
    )
    selection = empty.selections[0]
    assert selection.kernel_id == "portable.linear_s8.v1"
    assert selection.selection_basis == "measured_portable_fallback"
    assert selection.workload_key == workload
    assert selection.matched_cost is None
    assert "no exact measured latency cost" in selection.reason

    stale = _measured_cost(
        "optimized.linear_oi2.v1",
        workload,
        1,
        "stale-toolchain.json",
        toolchain="arm-none-eabi-stale",
    )
    stale_target = replace(CORTEX_M4, measured_costs=(stale,))
    stale_backend = select_backend_plan(
        plan,
        bakenn.CBackendOptions(
            kernel_policy=bakenn.KernelPolicy.MEASURED,
            target=stale_target,
        ),
    )
    assert stale_backend.selections[0].kernel_id == "portable.linear_s8.v1"
    assert stale_backend.selections[0].matched_cost is None
    assert "no exact measured latency cost" in stale_backend.selections[0].rejected[
        "optimized.linear_oi2.v1"
    ]


def test_measured_ties_prefer_portable_deterministically() -> None:
    plan = lower_to_plan(linear_graph())
    workload = canonical_workload_key(plan, plan.steps[0])
    portable_cost = _measured_cost(
        "portable.linear_s8.v1", workload, 100, "portable-tie.json"
    )
    optimized_cost = _measured_cost(
        "optimized.linear_oi2.v1", workload, 100, "optimized-tie.json"
    )
    target = replace(CORTEX_M4, measured_costs=(optimized_cost, portable_cost))
    backend = select_backend_plan(
        plan,
        bakenn.CBackendOptions(
            kernel_policy=bakenn.KernelPolicy.MEASURED,
            target=target,
        ),
    )
    assert backend.selections[0].kernel_id == "portable.linear_s8.v1"
    assert backend.selections[0].matched_cost == portable_cost


def test_static_priority_falls_back_for_small_linear_and_require_optimized_fails_closed() -> None:
    plan = lower_to_plan(linear_graph(3, 5))
    selected = select_backend_plan(
        plan, bakenn.CBackendOptions(kernel_policy=bakenn.KernelPolicy.STATIC_PRIORITY)
    )
    assert selected.selections[0].kernel_id == "portable.linear_s8.v1"
    assert "optimized.linear_oi2.v1" in selected.selections[0].rejected
    assert "optimized.linear_oi2_tail.v1" in selected.selections[0].rejected
    with pytest.raises(CompileError, match="no supported implementation"):
        select_backend_plan(
            plan,
            bakenn.CBackendOptions(kernel_policy=bakenn.KernelPolicy.REQUIRE_OPTIMIZED),
        )


def test_static_priority_selects_linear_tail_and_require_optimized_accepts_it() -> None:
    plan = lower_to_plan(linear_graph(12, 5))
    selected = select_backend_plan(
        plan, bakenn.CBackendOptions(kernel_policy=bakenn.KernelPolicy.STATIC_PRIORITY)
    )
    assert selected.selections[0].kernel_id == "optimized.linear_oi2_tail.v1"
    packed = selected.packed_constants["weight.linear_oi2_tail"]
    expected_pairs = plan.constants["weight"][:4].reshape(2, 2, 12).transpose(0, 2, 1)
    expected = np.concatenate((expected_pairs.reshape(-1), plan.constants["weight"][4]))
    np.testing.assert_array_equal(packed.value, expected)
    required = select_backend_plan(
        plan, bakenn.CBackendOptions(kernel_policy=bakenn.KernelPolicy.REQUIRE_OPTIMIZED)
    )
    assert required.selections[0].kernel_id == "optimized.linear_oi2_tail.v1"


def test_disabling_packing_makes_optimized_kernel_inapplicable() -> None:
    plan = lower_to_plan(linear_graph())
    backend = select_backend_plan(
        plan,
        bakenn.CBackendOptions(
            kernel_policy=bakenn.KernelPolicy.STATIC_PRIORITY,
            enable_weight_packing=False,
        ),
    )
    assert backend.selections[0].kernel_id == "portable.linear_s8.v1"
    assert backend.selections[0].rejected["optimized.linear_oi2.v1"] == (
        "weight packing is disabled"
    )


def test_manifest_records_reproducible_backend_decisions(tmp_path) -> None:
    compiled = bakenn.compile(
        linear_graph(),
        tmp_path,
        backend_options=bakenn.CBackendOptions(kernel_policy=bakenn.KernelPolicy.STATIC_PRIORITY),
    )
    manifest = json.loads(compiled.artifacts.manifest.read_text(encoding="utf-8"))
    operation = manifest["operations"][0]
    selection = manifest["backend"]["selections"][0]
    assert manifest["schema_version"] == 4
    assert manifest["backend"]["kernel_policy"] == "static_priority"
    assert manifest["backend"]["name"] == "c11"
    assert manifest["backend"]["optimized_steps"] == 1
    assert manifest["backend"]["weight_packing"] is True
    assert operation["kind"] == "linear_s8"
    assert selection["step_index"] == 0
    assert selection["step_name"] == "linear"
    assert selection["implementation"] == "optimized.linear_oi2.v1"
    assert selection["optimized"] is True
    assert selection["packed_constants"] == [
        {
            "alignment": 1,
            "bytes": 72,
            "layout": "linear_oi2_interleaved_v1",
            "name": "weight.linear_oi2",
            "source": "weight",
            "symbol": "bknn_p2_linear_packed_0",
        }
    ]
    assert manifest["constant_payload_bytes"] == manifest["constant_bytes"]
    assert manifest["constant_max_alignment"] == 1
    weights = compiled.artifacts.weights_source.read_text(encoding="utf-8")
    assert "_packed_0" in weights
    assert weights.count("const int8_t ") == 1
    assert "const int8_t bknn_p2_linear_constant_" not in weights


def test_backend_scratch_is_reusable_and_extends_arena_without_mutating_plan() -> None:
    plan = lower_to_plan(linear_graph())
    assert plan.scratch_size == 0
    assert plan.arena_size == 0
    selection = KernelSelection(
        step_index=0,
        step_name="linear",
        kernel_id="test.linear_with_scratch.v1",
        optimized=True,
        reason="test backend scratch contract",
        scratch_size=33,
        scratch_alignment=32,
    )
    backend = CBackendPlan(
        execution_plan=plan,
        options=bakenn.CBackendOptions(),
        selections=(selection,),
        packed_constants={},
    )
    assert backend.activation_arena_size == 0
    assert backend.scratch_offset == 0
    assert backend.scratch_size == 33
    assert backend.scratch_alignment == 32
    assert backend.arena_alignment == 32
    assert backend.arena_size == 64
    assert plan.scratch_offset is None
    assert plan.arena_size == 0
    context = StepEmitContext(
        plan,
        "bknn_scratch",
        0,
        {},
        selection,
        {},
        backend.scratch_offset,
    )
    assert context.scratch_pointer == "(void *)(arena + 0u)"


def test_packed_override_must_come_from_the_overridden_semantic_constant() -> None:
    packed = PackedConstant(
        name="wrong.packed",
        source="bias",
        layout="diagnostic_wrong_source_v1",
        value=np.zeros(6, dtype=np.int32),
    )
    with pytest.raises(ValueError, match="source must match"):
        KernelCapability(
            kernel_id="optimized.wrong_source.v1",
            priority=100,
            optimized=True,
            supported=True,
            reason="malformed diagnostic candidate",
            packed_constants=(packed,),
            constant_overrides={"weight": packed.name},
        )

    unrelated = PackedConstant(
        name="bias.packed",
        source="bias",
        layout="diagnostic_bias_v1",
        value=np.zeros(6, dtype=np.int32),
    )
    selection = KernelSelection(
        step_index=0,
        step_name="linear",
        kernel_id="test.non_step_source.v1",
        optimized=True,
        reason="malformed direct backend plan",
        packed_constants=(unrelated,),
    )
    plan = lower_to_plan(linear_graph())
    malformed_step = replace(plan.steps[0], bias="weight")
    malformed_plan = replace(plan, steps=(malformed_step,))
    with pytest.raises(CompileError, match="packs non-step constants"):
        CBackendPlan(
            execution_plan=malformed_plan,
            options=bakenn.CBackendOptions(),
            selections=(selection,),
            packed_constants={unrelated.name: unrelated},
        )


def test_backend_memory_requests_must_fit_the_32_bit_target_abi() -> None:
    with pytest.raises(ValueError, match="32-bit target byte range"):
        KernelCapability(
            kernel_id="optimized.oversized_scratch.v1",
            priority=100,
            optimized=True,
            supported=True,
            reason="malformed diagnostic candidate",
            scratch_size=TARGET_SIZE_MAX + 1,
        )
    with pytest.raises(ValueError, match="positive power of two"):
        PackedConstant(
            name="oversized_alignment.packed",
            source="weight",
            layout="oversized_alignment_v1",
            value=np.zeros(1, dtype=np.int8),
            alignment=1 << 32,
        )

    base = lower_to_plan(linear_graph())
    nearly_full = replace(
        base,
        activation_arena_size=TARGET_SIZE_MAX,
        arena_size=TARGET_SIZE_MAX,
    )
    selection = KernelSelection(
        step_index=0,
        step_name="linear",
        kernel_id="test.one_byte_scratch.v1",
        optimized=True,
        reason="forces aligned arena end beyond uint32",
        scratch_size=1,
    )
    with pytest.raises(CompileError, match="32-bit target byte limit"):
        CBackendPlan(
            execution_plan=nearly_full,
            options=bakenn.CBackendOptions(),
            selections=(selection,),
            packed_constants={},
        )
