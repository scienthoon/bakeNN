from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

import bakenn
from bakenn.errors import CompileError
from bakenn.targets import (
    CORTEX_M0PLUS,
    CORTEX_M4,
    ESP32,
    ESP32_C3,
    ESP32_S3,
    RV32IMC,
    KernelCostMeasurement,
    TargetArchitecture,
    TargetDescriptor,
    build_freestanding_elf,
    export_esp_idf_project,
)


def _graph() -> object:
    model = bakenn.FloatMLP(
        (
            bakenn.FloatLinear(
                np.asarray(
                    [
                        [1.0, -0.5, 0.25, 0.0],
                        [-0.75, 0.25, 0.5, 1.0],
                        [0.125, 0.5, -1.0, 0.25],
                    ],
                    dtype=np.float32,
                ),
                np.asarray([0.125, -0.25, 0.0625], dtype=np.float32),
                True,
                "hidden",
            ),
            bakenn.FloatLinear(
                np.asarray([[0.5, -1.0, 0.25], [-0.25, 0.75, 1.0]], dtype=np.float32),
                np.asarray([0.0, 0.125], dtype=np.float32),
                False,
                "output",
            ),
        ),
        "target_smoke",
    )
    calibration = np.asarray(
        [[-1.0, 0.0, 0.5, 1.0], [1.0, -0.5, 0.0, -1.0]], dtype=np.float32
    )
    return bakenn.quantize_ptq(model, calibration)


def test_target_profiles_are_explicit_and_have_no_fake_costs() -> None:
    assert set(bakenn.TARGET_PROFILES) == {
        "portable32",
        "cortex-m0plus",
        "cortex-m4",
        "rv32imc",
        "esp32",
        "esp32s3",
        "esp32c3",
    }
    assert CORTEX_M4.architecture is TargetArchitecture.ARM
    assert RV32IMC.architecture is TargetArchitecture.RISCV
    assert ESP32_S3.architecture is TargetArchitecture.XTENSA
    for descriptor in bakenn.TARGET_PROFILES.values():
        assert not descriptor.has_measured_costs
        assert descriptor.manifest()["measured_cost_table"] == {
            "available": False,
            "entry_count": 0,
            "entries": [],
        }


def test_target_descriptor_and_measurement_validation() -> None:
    with pytest.raises(ValueError, match="power of two"):
        replace(CORTEX_M4, arena_alignment=3)
    with pytest.raises(ValueError, match="positive integer"):
        replace(CORTEX_M4, flash_bytes=0)
    with pytest.raises(ValueError, match="positive integer"):
        replace(CORTEX_M4, sram_bytes=1 << 32)
    with pytest.raises(ValueError, match="positive integer"):
        KernelCostMeasurement("kernel", "shape", 0, "gcc", ("-O2",), "raw.csv")
    measured = KernelCostMeasurement(
        "portable.linear.v1", "1x32x16", 1234, "arm-none-eabi-gcc 14", ("-O2",), "run-1.json"
    )
    descriptor = replace(CORTEX_M4, measured_costs=(measured,))
    assert descriptor.has_measured_costs
    assert descriptor.manifest()["measured_cost_table"] == {
        "available": True,
        "entry_count": 1,
        "entries": [measured.manifest()],
    }
    with pytest.raises(ValueError, match="unique"):
        replace(CORTEX_M4, measured_costs=(measured, measured))


def test_declared_resource_budgets_fail_on_known_lower_bounds(tmp_path: Path) -> None:
    with pytest.raises(CompileError, match="constant payload"):
        bakenn.compile(
            _graph(),
            tmp_path / "flash",
            target=replace(CORTEX_M4, flash_bytes=1),
        )
    with pytest.raises(CompileError, match="arena"):
        bakenn.compile(
            _graph(),
            tmp_path / "sram",
            target=replace(CORTEX_M4, sram_bytes=1),
        )


def test_target_argument_cannot_silently_override_backend_target(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="conflicts"):
        bakenn.compile(
            _graph(),
            tmp_path / "conflict",
            target=CORTEX_M4,
            backend_options=bakenn.CBackendOptions(target=RV32IMC),
        )


@pytest.mark.parametrize(
    ("target", "expected_arena_alignment", "expected_constant_alignment"),
    [
        ("portable32", 1, 1),
        ("cortex-m0plus", 4, 4),
        ("cortex-m4", 8, 4),
        ("rv32imc", 4, 4),
        ("esp32", 4, 4),
        ("esp32s3", 16, 16),
        ("esp32c3", 4, 4),
    ],
)
def test_compile_records_target_and_applies_storage_alignment(
    tmp_path: Path,
    target: str,
    expected_arena_alignment: int,
    expected_constant_alignment: int,
) -> None:
    compiled = bakenn.compile(_graph(), tmp_path / target, target=target)
    manifest = json.loads(compiled.artifacts.manifest.read_text(encoding="utf-8"))
    assert manifest["backend"]["target"]["id"] == target
    assert manifest["arena_alignment"] >= expected_arena_alignment
    assert manifest["constant_max_alignment"] >= expected_constant_alignment
    weights = compiled.artifacts.weights_source.read_text(encoding="utf-8")
    if expected_constant_alignment > 1:
        assert f"_Alignas({expected_constant_alignment})" in weights


@pytest.mark.parametrize("target", [ESP32, ESP32_S3, ESP32_C3])
def test_esp_idf_project_is_self_contained_and_target_checked(
    tmp_path: Path, target: TargetDescriptor
) -> None:
    compiled = bakenn.compile(_graph(), tmp_path / "generated", target=target)
    project = export_esp_idf_project(compiled.artifacts, target, tmp_path / "idf")
    assert project.target is target
    assert (project.root / "CMakeLists.txt").is_file()
    assert (project.root / "sdkconfig.defaults").is_file()
    assert (project.component / compiled.artifacts.header.name).is_file()
    assert "idf_component_register" in (project.component / "CMakeLists.txt").read_text()
    runner = (project.main / "main.c").read_text(encoding="utf-8")
    assert "esp_cpu_get_cycle_count" in runner
    assert "uxTaskGetStackHighWaterMark" in runner
    assert "BAKENN_OUTPUT" in runner
    target_data = json.loads((project.root / "bakenn_target.json").read_text())
    assert target_data["idf_target"] == target.metadata["idf_target"]

    wrong = ESP32_C3 if target is not ESP32_C3 else ESP32_S3
    with pytest.raises(CompileError, match="does not match"):
        export_esp_idf_project(compiled.artifacts, wrong, tmp_path / "wrong")


@pytest.mark.parametrize(
    ("target", "candidates", "require_variable"),
    [
        (CORTEX_M0PLUS, ("arm-none-eabi-gcc",), "BAKENN_REQUIRE_ARM_CC"),
        (CORTEX_M4, ("arm-none-eabi-gcc",), "BAKENN_REQUIRE_ARM_CC"),
        (
            RV32IMC,
            (
                "riscv-none-elf-gcc",
                "riscv64-elf-gcc",
                "riscv64-unknown-elf-gcc",
                "riscv32-unknown-elf-gcc",
            ),
            "BAKENN_REQUIRE_RISCV_CC",
        ),
    ],
)
def test_freestanding_cross_link_when_toolchain_is_available(
    tmp_path: Path,
    target: TargetDescriptor,
    candidates: tuple[str, ...],
    require_variable: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiler = next((value for value in candidates if shutil.which(value)), None)
    if compiler is None:
        if __import__("os").environ.get(require_variable) == "1":
            pytest.fail(f"required cross compiler is missing: {', '.join(candidates)}")
        pytest.skip(f"cross compiler not installed: {', '.join(candidates)}")
    compiled = bakenn.compile(_graph(), tmp_path / "generated", target=target)
    report = build_freestanding_elf(
        compiled.artifacts,
        target,
        tmp_path / "cross",
        compiler=compiler,
    )
    assert report.elf.is_file()
    assert report.map_file.is_file()
    assert report.flash_load_bytes > 0
    assert report.static_sram_bytes >= report.model_arena_bytes
    assert report.undefined_symbols == ()
    assert report.forbidden_symbols == ()
    assert (tmp_path / "cross" / f"bknn_target_smoke_{target.target_id}_report.json").is_file()


def test_freestanding_build_rejects_wrong_artifact_target(tmp_path: Path) -> None:
    compiled = bakenn.compile(_graph(), tmp_path / "generated", target=CORTEX_M4)
    with pytest.raises(CompileError, match="does not match"):
        build_freestanding_elf(compiled.artifacts, RV32IMC, tmp_path / "cross")
