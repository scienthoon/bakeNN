from __future__ import annotations

import json
from pathlib import Path
import zipfile

from scripts.build_release_evidence import build_archive


def test_release_evidence_is_deterministic_and_manifested(tmp_path: Path) -> None:
    artifact = tmp_path / "target.elf"
    artifact.write_bytes(b"exact physical artifact\x00")
    first, first_digest = build_archive(
        tmp_path / "first.zip", (f"nrf52840/firmware.elf={artifact}",)
    )
    second, second_digest = build_archive(
        tmp_path / "second.zip", (f"nrf52840/firmware.elf={artifact}",)
    )
    assert first_digest == second_digest
    assert first.read_bytes() == second.read_bytes()

    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
        assert names == sorted(names)
        assert "artifacts/nrf52840/firmware.elf" in names
        manifest = json.loads(archive.read("MANIFEST.json"))
        assert manifest["bakenn_version"] == "0.1.0"
        assert len(manifest["source_commit"]) == 40
        assert manifest["external_artifacts"][0]["bytes"] == 24
        assert any(
            item["path"].endswith("iotlab_447626_direct_cmsis_fc_uart.txt")
            for item in manifest["files"]
        )
        assert any(
            item["path"] == "examples/mnist/evidence/mnist_fp32.pt"
            for item in manifest["files"]
        )
        assert any(
            item["path"]
            == "benchmarks/cross_build/results/mnist_esp32_cross_build.json"
            for item in manifest["files"]
        )
        assert any(
            item["path"]
            == "benchmarks/microtvm_compare/results/mnist_cortex_m4_cross_build.json"
            for item in manifest["files"]
        )


def test_release_evidence_rejects_unsafe_artifact_label(tmp_path: Path) -> None:
    artifact = tmp_path / "target.elf"
    artifact.write_bytes(b"x")
    try:
        build_archive(tmp_path / "bad.zip", (f"../escape={artifact}",))
    except ValueError as error:
        assert "unsafe artifact label" in str(error)
    else:
        raise AssertionError("unsafe label was accepted")
