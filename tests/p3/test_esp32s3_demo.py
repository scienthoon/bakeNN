from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


pytest.importorskip("torch")


def test_fp32_demo_generates_self_contained_s3_project(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "demo"
    subprocess.run(
        [
            sys.executable,
            str(root / "examples/esp32s3_end_to_end/generate.py"),
            "--output",
            str(output),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((output / "demo_summary.json").read_text())
    kernel_ids = {item["kernel"] for item in summary["selections"]}
    assert any("esp_nn.esp32s3.conv2d_s8" in value for value in kernel_ids)
    assert any("esp_nn.esp32s3.depthwise_conv2d_s8" in value for value in kernel_ids)
    assert any("esp_nn.esp32s3.linear_per_channel_s8" in value for value in kernel_ids)
    project = output / "esp_idf"
    assert (project / "CMakeLists.txt").is_file()
    assert (project / "components/bakenn_model/CMakeLists.txt").is_file()
    runner = (project / "main/main.c").read_text(encoding="utf-8")
    assert "median_cycles=" in runner
    assert "BAKENN_OUTPUT_FNV1A" in runner
