from __future__ import annotations

from dataclasses import replace
import json

import bakenn
from bakenn.targets import CORTEX_M4
from tests.p0.model_fixtures import mobilenet_v1_graph, residual_ds_cnn_graph, tiny_cnn_graph


def test_memory_report_is_generated_deterministically_and_matches_manifest(tmp_path) -> None:
    graph = mobilenet_v1_graph()
    first = bakenn.compile(graph, tmp_path / "first")
    second = bakenn.compile(graph, tmp_path / "second")

    assert first.memory_report is first.artifacts.memory_report
    assert first.artifacts.memory_report_json.is_file()
    assert first.artifacts.memory_report_text.is_file()
    assert first.artifacts.memory_report_json.read_bytes() == second.artifacts.memory_report_json.read_bytes()
    assert first.artifacts.memory_report_text.read_bytes() == second.artifacts.memory_report_text.read_bytes()

    report = json.loads(first.artifacts.memory_report_json.read_text(encoding="utf-8"))
    manifest = json.loads(first.artifacts.manifest.read_text(encoding="utf-8"))
    compile_time = report["compile_time"]
    assert report["schema_version"] == 1
    assert report["model"] == graph.name
    assert report["target"] == "portable32"
    assert compile_time["activation_arena_bytes"] == first.plan.activation_arena_size
    assert compile_time["scratch_bytes"] == first.artifacts.backend_plan.scratch_size
    assert compile_time["arena_bytes"] == first.artifacts.backend_plan.arena_size
    assert compile_time["emitted_constant_payload_bytes"] == manifest["constant_payload_bytes"]
    assert compile_time["semantic_constant_bytes"] == sum(
        value.nbytes for value in first.plan.constants.values()
    )
    assert compile_time["generated_model_heap_calls"] == 0
    assert manifest["memory_report"] == {
        "json": first.artifacts.memory_report_json.name,
        "schema_version": 1,
        "text": first.artifacts.memory_report_text.name,
    }

    text = first.artifacts.memory_report_text.read_text(encoding="utf-8")
    assert "BakeNN Memory Report" in text
    assert "Compile-time exact" in text
    assert "Caller input/output" in text
    assert "Generated-model heap     0 calls" in text
    assert "Not measured at AOT compile time" in text


def test_memory_report_exposes_peak_liveness_and_physical_reuse(tmp_path) -> None:
    residual = bakenn.compile(residual_ds_cnn_graph(), tmp_path / "residual").memory_report
    assert residual.peak_working_payload_bytes == 96
    assert tuple(residual.steps[index].name for index in residual.peak_step_indices) == (
        "project",
    )
    peak = residual.steps[residual.peak_step_indices[0]]
    assert peak.live_buffers == (
        "block.residual",
        "depthwise.output",
        "project.output",
    )
    assert peak.live_activation_payload_bytes == 96
    assert peak.scratch_bytes == 0

    mobile_compiled = bakenn.compile(mobilenet_v1_graph(), tmp_path / "mobile")
    mobile = mobile_compiled.memory_report
    reused = next(
        region
        for region in mobile.reuse_regions
        if region.buffers == ("depthwise.output", "pool.output")
    )
    assert reused.offset == 0
    assert reused.size_bytes == 3
    buffers = {item.name: item for item in mobile.buffers}
    for name, lifetime in mobile_compiled.plan.lifetimes.items():
        assert buffers[name].birth_step == lifetime.birth
        assert buffers[name].death_step_exclusive == lifetime.death
    assert buffers["depthwise.output"].death_step_exclusive <= buffers["pool.output"].birth_step
    assert buffers["pool.output"].members == ("pool.output", "flatten.output")


def test_memory_report_uses_selected_backend_scratch_and_marks_measurement_boundaries(tmp_path) -> None:
    target = replace(CORTEX_M4, flash_bytes=64 * 1024, sram_bytes=8 * 1024)
    compiled = bakenn.compile(
        tiny_cnn_graph(),
        tmp_path / "cortex_m4",
        backend_options=bakenn.CBackendOptions(
            kernel_policy=bakenn.KernelPolicy.AUTO,
            target=target,
        ),
    )
    report = compiled.memory_report
    conv = report.steps[0]
    assert conv.kernel_id == "cortex_m4.conv2d_3x3_im2col_smlad.v1"
    assert conv.optimized is True
    assert conv.scratch_bytes == 20
    assert report.scratch_bytes == 20
    assert report.arena_bytes > report.activation_arena_bytes

    data = report.json_data()
    budgets = data["target_budgets"]
    assert budgets["flash_bytes"] == 64 * 1024
    assert budgets["constant_payload_headroom_bytes"] == (
        64 * 1024 - report.emitted_constant_payload_bytes
    )
    assert budgets["sram_bytes"] == 8 * 1024
    assert budgets["arena_headroom_bytes"] == 8 * 1024 - report.arena_bytes
    unknown = data["not_measured_at_aot"]
    assert unknown["final_flash_load_bytes"]["value"] is None
    assert "ELF/map" in unknown["final_flash_load_bytes"]["reason"]
    assert unknown["full_firmware_peak_sram_bytes"]["value"] is None
    assert unknown["peak_stack_bytes"]["value"] is None

    text = report.to_text()
    assert "Declared target budgets (lower-bound checks)" in text
    assert "code/data excluded" in text
    assert "caller I/O/app/stack excluded" in text
    assert "selected-kernel scratch: 20 B" in text
