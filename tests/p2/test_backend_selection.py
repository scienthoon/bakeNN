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


def test_portable_is_default_and_auto_selects_deterministic_packed_linear() -> None:
    plan = lower_to_plan(linear_graph())
    portable = select_backend_plan(plan)
    assert portable.selections[0].kernel_id == "portable.linear_s8.v1"
    assert not portable.selections[0].optimized
    assert not portable.packed_constants

    options = bakenn.CBackendOptions(kernel_policy=bakenn.KernelPolicy.AUTO)
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


def test_auto_falls_back_for_small_linear_and_require_optimized_fails_closed() -> None:
    plan = lower_to_plan(linear_graph(3, 5))
    auto = select_backend_plan(
        plan, bakenn.CBackendOptions(kernel_policy=bakenn.KernelPolicy.AUTO)
    )
    assert auto.selections[0].kernel_id == "portable.linear_s8.v1"
    assert "optimized.linear_oi2.v1" in auto.selections[0].rejected
    assert "optimized.linear_oi2_tail.v1" in auto.selections[0].rejected
    with pytest.raises(CompileError, match="no supported implementation"):
        select_backend_plan(
            plan,
            bakenn.CBackendOptions(kernel_policy=bakenn.KernelPolicy.REQUIRE_OPTIMIZED),
        )


def test_auto_selects_linear_tail_and_require_optimized_accepts_it() -> None:
    plan = lower_to_plan(linear_graph(12, 5))
    auto = select_backend_plan(
        plan, bakenn.CBackendOptions(kernel_policy=bakenn.KernelPolicy.AUTO)
    )
    assert auto.selections[0].kernel_id == "optimized.linear_oi2_tail.v1"
    packed = auto.packed_constants["weight.linear_oi2_tail"]
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
            kernel_policy=bakenn.KernelPolicy.AUTO,
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
        backend_options=bakenn.CBackendOptions(kernel_policy=bakenn.KernelPolicy.AUTO),
    )
    manifest = json.loads(compiled.artifacts.manifest.read_text(encoding="utf-8"))
    operation = manifest["operations"][0]
    selection = manifest["backend"]["selections"][0]
    assert manifest["schema_version"] == 3
    assert manifest["backend"]["kernel_policy"] == "auto"
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
