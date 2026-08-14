from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import bakenn
from bakenn.errors import CompileError


def _graph() -> object:
    model = bakenn.FloatMLP(
        (
            bakenn.FloatLinear(
                np.asarray([[1.0, -0.5, 0.25, 0.0], [-0.75, 0.25, 0.5, 1.0]], dtype=np.float32),
                np.asarray([0.125, -0.25], dtype=np.float32),
                False,
                "output",
            ),
        ),
        "zephyr_smoke",
    )
    calibration = np.asarray(
        [[-1.0, 0.0, 0.5, 1.0], [1.0, -0.5, 0.0, -1.0]], dtype=np.float32
    )
    return bakenn.quantize_ptq(model, calibration)


@pytest.mark.parametrize(
    ("board", "iotlab_architecture"),
    [
        ("nrf52840dk_nrf52840", "nrf52840dk"),
        ("nrf52dk_nrf52832", "nrf52dk"),
        ("disco_l475_iot1", "st-iotnode"),
    ],
)
def test_zephyr_project_is_self_contained_and_records_measurement_contract(
    tmp_path: Path, board: str, iotlab_architecture: str
) -> None:
    compiled = bakenn.compile(_graph(), tmp_path / "generated", target="cortex-m4")
    project = bakenn.export_zephyr_project(
        compiled.artifacts, "cortex-m4", tmp_path / board, board=board
    )
    assert project.board == board
    assert project.iotlab_architecture == iotlab_architecture
    assert (project.root / "CMakeLists.txt").is_file()
    assert (project.root / "prj.conf").is_file()
    assert (project.generated / compiled.artifacts.header.name).is_file()
    main = (project.source / "main.c").read_text(encoding="utf-8")
    assert "timing_cycles_get" in main
    assert "__has_include(<zephyr/kernel.h>)" in main
    assert "#include <kernel.h>" in main
    assert "k_thread_stack_space_get" in main
    assert "BAKENN_OUTPUT" in main
    assert "BAKENN_BENCHMARK_RUNS 101u" in main
    assert "INPUT_ZERO_POINT" in main
    metadata = json.loads((project.root / "bakenn_target.json").read_text())
    assert metadata["zephyr_board"] == board
    assert metadata["iotlab_architecture"] == iotlab_architecture
    assert metadata["benchmark"]["measured_runs"] == 101


def test_zephyr_export_fails_closed_for_wrong_target_or_board(tmp_path: Path) -> None:
    m4 = bakenn.compile(_graph(), tmp_path / "m4", target="cortex-m4")
    with pytest.raises(CompileError, match="unsupported Zephyr"):
        bakenn.export_zephyr_project(
            m4.artifacts, "cortex-m4", tmp_path / "bad-board", board="unknown"
        )
    with pytest.raises(CompileError, match="requires the cortex-m4"):
        bakenn.export_zephyr_project(
            m4.artifacts, "cortex-m0plus", tmp_path / "wrong-target"
        )
    portable = bakenn.compile(_graph(), tmp_path / "portable", target="portable32")
    with pytest.raises(CompileError, match="artifact target"):
        bakenn.export_zephyr_project(
            portable.artifacts, "cortex-m4", tmp_path / "wrong-artifact"
        )
