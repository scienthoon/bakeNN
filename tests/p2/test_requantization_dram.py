"""Opt-in ESP-IDF DRAM placement of per-channel requantization arrays."""

from __future__ import annotations

from dataclasses import replace
import json
import re
import subprocess
from pathlib import Path

import pytest

import bakenn
from bakenn.artifacts import manifest_payload_sha256, validate_manifest
from bakenn.backend.portable_c.selection import requantization_array_bytes
from bakenn.errors import CompileError
from bakenn.targets import CORTEX_M4, ESP32, ESP32_C3, ESP32_S3, PORTABLE_32, export_esp_idf_project
from tests.p0.model_fixtures import residual_ds_cnn_graph
from tests.p2.test_esp_nn import _compare, _runner_source

from .support import require_compiler


_REQUANTIZATION_DEFINITION = re.compile(
    r"^(DRAM_ATTR )?const int32_t (\w+_op\d+_(?:multiplier|shift))\[(\d+)\] = \{",
    re.MULTILINE,
)


def _options(target, *, dram: bool, **overrides):  # type: ignore[no-untyped-def]
    return bakenn.CBackendOptions(
        kernel_policy=bakenn.KernelPolicy.STATIC_PRIORITY,
        enable_esp_nn=target.target_id in {"esp32", "esp32s3"},
        target=target,
        requantization_in_dram=dram,
        **overrides,
    )


def _compile(tmp_path: Path, target, *, dram: bool, name: str = "model"):  # type: ignore[no-untyped-def]
    # One C symbol per target so flash and DRAM variants are directly comparable.
    return bakenn.compile(
        residual_ds_cnn_graph(),
        tmp_path / name,
        model_name=f"dram_{target.target_id}",
        backend_options=_options(target, dram=dram),
        target=target,
    )


def test_requantization_in_dram_option_requires_boolean() -> None:
    with pytest.raises(ValueError, match="requantization_in_dram"):
        bakenn.CBackendOptions(requantization_in_dram=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("target", [PORTABLE_32, CORTEX_M4], ids=lambda item: item.target_id)
def test_requantization_in_dram_requires_an_esp_idf_target(tmp_path: Path, target) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(CompileError, match="requires an ESP-IDF target"):
        _compile(tmp_path, target, dram=True)
    assert not (tmp_path / "model").exists()


@pytest.mark.parametrize("target", [ESP32, ESP32_S3, ESP32_C3], ids=lambda item: item.target_id)
def test_requantization_in_dram_moves_only_multiplier_and_shift_arrays(
    tmp_path: Path,
    target,  # type: ignore[no-untyped-def]
) -> None:
    default = _compile(tmp_path, target, dram=False, name="flash")
    placed = _compile(tmp_path, target, dram=True, name="dram")
    default_weights = default.artifacts.weights_source.read_text(encoding="utf-8")
    placed_weights = placed.artifacts.weights_source.read_text(encoding="utf-8")

    default_arrays = _REQUANTIZATION_DEFINITION.findall(default_weights)
    placed_arrays = _REQUANTIZATION_DEFINITION.findall(placed_weights)
    assert default_arrays and all(prefix == "" for prefix, _, _ in default_arrays)
    assert placed_arrays and all(prefix == "DRAM_ATTR " for prefix, _, _ in placed_arrays)
    assert placed_weights.count("DRAM_ATTR ") == len(placed_arrays)
    assert '#include "esp_attr.h"' in placed_weights
    assert "esp_attr.h" not in default_weights
    # Stripping the placement recovers the default translation unit exactly,
    # and every other generated source is unchanged.
    stripped = placed_weights.replace('#include "esp_attr.h"\n', "").replace("DRAM_ATTR ", "")
    assert stripped == default_weights
    for name in ("header", "model_source", "weights_header", "kernels_header", "kernels_source"):
        assert getattr(placed.artifacts, name).read_bytes() == getattr(
            default.artifacts, name
        ).read_bytes(), name

    placed_bytes = sum(4 * int(count) for _, _, count in placed_arrays)
    assert placed_bytes == requantization_array_bytes(placed.plan)
    assert placed.artifacts.backend_plan.arena_size == default.artifacts.backend_plan.arena_size
    assert [item.kernel_id for item in placed.artifacts.backend_plan.selections] == [
        item.kernel_id for item in default.artifacts.backend_plan.selections
    ]

    manifest = json.loads(placed.artifacts.manifest.read_text(encoding="utf-8"))
    assert manifest["backend"]["requantization_placement"] == {
        "memory": "dram",
        "attribute": "DRAM_ATTR",
        "header": "esp_attr.h",
        "bytes": placed_bytes,
        "symbols": sorted(symbol for _, symbol, _ in placed_arrays),
    }
    default_manifest = json.loads(default.artifacts.manifest.read_text(encoding="utf-8"))
    assert "requantization_placement" not in default_manifest["backend"]
    assert manifest["constant_payload_bytes"] == default_manifest["constant_payload_bytes"]
    bakenn.load_manifest(placed.artifacts.manifest)

    report = json.loads(placed.artifacts.memory_report_json.read_text(encoding="utf-8"))
    assert report["compile_time"]["dram_resident_constant_bytes"] == placed_bytes
    default_report = json.loads(default.artifacts.memory_report_json.read_text(encoding="utf-8"))
    assert "dram_resident_constant_bytes" not in default_report["compile_time"]
    assert f"DRAM-resident constants  {placed_bytes} B" in placed.memory_report.to_text()
    assert "DRAM-resident" not in default.memory_report.to_text()


def test_requantization_in_dram_reserves_sram_before_kernel_selection(
    tmp_path: Path,
) -> None:
    unbounded = _compile(tmp_path, ESP32_S3, dram=True, name="unbounded")
    reserved = requantization_array_bytes(unbounded.plan)
    preferred_arena = unbounded.artifacts.backend_plan.arena_size
    assert any(
        item.kernel_id.startswith("esp_nn.") and item.scratch_size
        for item in unbounded.artifacts.backend_plan.selections
    )
    # The preferred ESP-NN arena fits on its own but not beside the DRAM arrays.
    budget = preferred_arena + reserved - 1
    target = replace(ESP32_S3, sram_bytes=budget)

    flash_resident = bakenn.compile(
        residual_ds_cnn_graph(),
        tmp_path / "flash_budget",
        backend_options=_options(target, dram=False),
        target=target,
    )
    dram_resident = bakenn.compile(
        residual_ds_cnn_graph(),
        tmp_path / "dram_budget",
        backend_options=_options(target, dram=True),
        target=target,
    )

    assert flash_resident.artifacts.backend_plan.arena_size == preferred_arena
    dram_plan = dram_resident.artifacts.backend_plan
    assert dram_plan.arena_size + reserved <= budget
    notes = [
        reason
        for item in dram_plan.selections
        for reason in item.rejected.values()
        if f"after reserving {reserved} bytes of DRAM-resident" in reason
    ]
    assert notes
    report = json.loads(dram_resident.artifacts.memory_report_json.read_text(encoding="utf-8"))
    assert report["target_budgets"]["sram_headroom_after_dram_constants_bytes"] == (
        budget - dram_plan.arena_size - reserved
    )

    too_small = replace(ESP32_S3, sram_bytes=reserved - 1)
    with pytest.raises(CompileError, match="DRAM-resident requantization constants"):
        bakenn.compile(
            residual_ds_cnn_graph(),
            tmp_path / "too_small",
            backend_options=_options(too_small, dram=True),
            target=too_small,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda placement: placement.update(memory="iram"), "memory must be 'dram'"),
        (lambda placement: placement.update(attribute="IRAM_ATTR"), "attribute must be 'DRAM_ATTR'"),
        (lambda placement: placement.update(bytes=1 << 30), "cannot exceed constant_payload_bytes"),
        (lambda placement: placement["symbols"].reverse(), "sorted and unique"),
        (lambda placement: placement.update(extra=True), "unknown keys"),
    ],
)
def test_requantization_placement_manifest_is_strictly_validated(
    tmp_path: Path,
    mutation,  # type: ignore[no-untyped-def]
    message: str,
) -> None:
    compiled = _compile(tmp_path, ESP32_S3, dram=True)
    manifest = json.loads(compiled.artifacts.manifest.read_text(encoding="utf-8"))
    mutation(manifest["backend"]["requantization_placement"])
    manifest["manifest_payload_sha256"] = manifest_payload_sha256(manifest)

    with pytest.raises(CompileError, match=message):
        validate_manifest(manifest)


def test_dram_placed_esp32_artifact_runs_byte_exact_on_host(tmp_path: Path) -> None:
    compiler = require_compiler("clang")
    compiled = _compile(tmp_path, ESP32, dram=True)
    stub = tmp_path / "esp_idf_stub"
    stub.mkdir()
    # Stand-in for ESP-IDF's esp_attr.h: a real section attribute exercises the
    # same declaration syntax that DRAM_ATTR expands to on the target.
    (stub / "esp_attr.h").write_text(
        "#pragma once\n"
        "#if defined(__APPLE__)\n"
        '#define DRAM_ATTR __attribute__((section("__DATA,__bknn_dram")))\n'
        "#else\n"
        '#define DRAM_ATTR __attribute__((section(".dram1.bknn")))\n'
        "#endif\n",
        encoding="utf-8",
    )
    runner = tmp_path / "runner.c"
    runner.write_text(_runner_source(compiled), encoding="utf-8")
    executable = tmp_path / "runner"
    artifacts = compiled.artifacts
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-O2",
            "-DCONFIG_NN_OPTIMIZED=1",
            "-Wall",
            "-Wextra",
            "-Wno-unused-parameter",
            "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all",
            str(artifacts.model_source),
            str(artifacts.weights_source),
            str(artifacts.kernels_source),
            *(str(source) for source in artifacts.support_sources),
            str(runner),
            "-I",
            str(artifacts.output_dir),
            "-I",
            str(stub),
            *(flag for path in artifacts.support_include_dirs for flag in ("-I", str(path))),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
    )

    _compare(compiled, executable, count=64, seed=38)


def test_dram_placed_artifact_exports_as_esp_idf_project(tmp_path: Path) -> None:
    compiled = _compile(tmp_path, ESP32_S3, dram=True)
    project = export_esp_idf_project(compiled.artifacts, ESP32_S3, tmp_path / "esp_idf")
    exported = next((project.component / "generated").glob("*_weights.c"))
    text = exported.read_text(encoding="utf-8")
    assert '#include "esp_attr.h"' in text
    assert "DRAM_ATTR const int32_t" in text
