from __future__ import annotations

from pathlib import Path

import pytest

import bakenn
from bakenn.artifacts import load_manifest
from bakenn.errors import CompileError
from bakenn.targets import esp_idf, zephyr
from tests.p2.test_backend_selection import linear_graph


EXPORTERS = (
    (bakenn.export_esp_idf_component, "esp32"),
    (bakenn.export_esp_idf_project, "esp32"),
    (bakenn.export_zephyr_project, "cortex-m4"),
)


def _tree(root: Path) -> dict[str, bytes | None]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


@pytest.mark.parametrize(("exporter", "target"), EXPORTERS)
def test_export_preserves_existing_firmware(tmp_path, exporter, target):
    artifacts = bakenn.compile(linear_graph(), tmp_path / "model", target=target).artifacts
    output = tmp_path / "firmware"
    (output / "main").mkdir(parents=True)
    (output / "CMakeLists.txt").write_text("# User project configuration\n")
    (output / "main" / "main.c").write_text("/* User firmware */\n")
    before = _tree(output)

    with pytest.raises(CompileError, match="non-empty"):
        exporter(artifacts, target, output)

    assert _tree(output) == before
    load_manifest(artifacts.manifest)


@pytest.mark.parametrize(("exporter", "target"), EXPORTERS)
def test_export_rejects_nested_destination_before_touching_source(
    tmp_path, monkeypatch, exporter, target
):
    artifacts = bakenn.compile(linear_graph(), tmp_path / "model", target=target).artifacts
    before = _tree(artifacts.output_dir)

    # Do not allow the unfixed component exporter to recursively copy itself.
    def reject_recursive_copy(*args, **kwargs):
        raise AssertionError("copytree reached before checking source/destination overlap")

    monkeypatch.setattr(esp_idf.shutil, "copytree", reject_recursive_copy)
    with pytest.raises(CompileError, match="inside|overlap"):
        exporter(artifacts, target, artifacts.output_dir / "firmware")

    assert _tree(artifacts.output_dir) == before
    load_manifest(artifacts.manifest)


@pytest.mark.parametrize(("exporter", "target"), EXPORTERS)
def test_export_rejects_symlink_destination(tmp_path, exporter, target):
    artifacts = bakenn.compile(linear_graph(), tmp_path / "model", target=target).artifacts
    real = tmp_path / "existing"
    real.mkdir()
    output = tmp_path / "linked"
    output.symlink_to(real, target_is_directory=True)

    with pytest.raises(CompileError, match="symlink"):
        exporter(artifacts, target, output)

    assert list(real.iterdir()) == []
    assert output.is_symlink()


@pytest.mark.parametrize(("module", "exporter", "target"), (
    (esp_idf, bakenn.export_esp_idf_project, "esp32"),
    (zephyr, bakenn.export_zephyr_project, "cortex-m4"),
))
@pytest.mark.parametrize("existing_empty", (False, True))
def test_failed_export_leaves_no_partial_project(
    tmp_path, monkeypatch, module, exporter, target, existing_empty
):
    artifacts = bakenn.compile(linear_graph(), tmp_path / "model", target=target).artifacts
    output = tmp_path / "firmware"
    if existing_empty:
        output.mkdir()

    def fail_main(*args, **kwargs):
        raise RuntimeError("injected runner generation failure")

    monkeypatch.setattr(module, "_main_source", fail_main)
    with pytest.raises(RuntimeError, match="injected"):
        exporter(artifacts, target, output)

    assert output.exists() == existing_empty
    if existing_empty:
        assert list(output.iterdir()) == []
    assert list(tmp_path.glob(".firmware.bakenn-export-*")) == []
    load_manifest(artifacts.manifest)


@pytest.mark.parametrize(("exporter", "target"), EXPORTERS)
def test_export_to_empty_directory_publishes_valid_closure(tmp_path, exporter, target):
    artifacts = bakenn.compile(linear_graph(), tmp_path / "model", target=target).artifacts
    output = tmp_path / "firmware"
    output.mkdir()
    exporter(artifacts, target, output)
    copied = list(output.rglob(artifacts.manifest.name))
    assert len(copied) == 1
    load_manifest(copied[0])
    load_manifest(artifacts.manifest)
    assert list(tmp_path.glob(".firmware.bakenn-export-*")) == []
