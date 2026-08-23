from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess

import pytest

import bakenn
from bakenn.artifacts import (
    ARITHMETIC_PROFILE_VERSION,
    GENERATED_C_ABI_VERSION,
    MANIFEST_SCHEMA_VERSION,
    load_manifest,
    validate_manifest,
)
from bakenn.backend.portable_c import generator
from bakenn.backend.portable_c.selection import canonical_workload_key
from bakenn.errors import CompileError
from bakenn.plan import lower_to_plan
from bakenn.targets import CORTEX_M4, KernelCostMeasurement
from benchmarks.tflm_compare.model_fixtures import cmsis_mlp_graph
from tests.p0.model_fixtures import tiny_cnn_graph
from tests.p2.test_backend_selection import linear_graph


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _transaction_paths(parent: Path, name: str) -> list[Path]:
    return sorted(
        (
            *parent.glob(f".{name}.bakenn-staging-*"),
            *parent.glob(f".{name}.bakenn-backup-*"),
        )
    )


def test_public_header_is_strict_c11_and_cpp17_link_compatible(
    tmp_path: Path,
) -> None:
    cc = shutil.which("cc")
    cxx = shutil.which("c++")
    if cc is None or cxx is None:
        pytest.skip("C and C++ compilers are required for the generated ABI test")

    compiled = bakenn.compile(tiny_cnn_graph(), tmp_path / "generated")
    artifacts = compiled.artifacts
    manifest = load_manifest(artifacts.manifest)
    symbol = str(manifest["model"])
    macro = symbol.upper()
    header_text = artifacts.header.read_text(encoding="utf-8")
    assert "#define BKNN_LAYOUT_NLC 3u" in header_text
    assert 'extern "C" {' in header_text
    assert "BKNN_RESTRICT" in header_text
    assert "byte ranges are pairwise non-overlapping" in header_text
    assert f"{macro}_ARENA_ALIGNMENT bytes" in header_text

    build = tmp_path / "build"
    build.mkdir()
    objects: list[Path] = []
    for source in (
        artifacts.model_source,
        artifacts.weights_source,
        artifacts.kernels_source,
    ):
        object_path = build / f"{source.stem}.o"
        subprocess.run(
            [
                cc,
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-pedantic",
                "-I",
                str(artifacts.output_dir),
                "-c",
                str(source),
                "-o",
                str(object_path),
            ],
            check=True,
            capture_output=True,
        )
        objects.append(object_path)

    runner = build / "runner.cpp"
    runner.write_text(
        f'''#include "{artifacts.header.name}"
#include <cstdint>

static_assert(BKNN_C_ABI_VERSION == {GENERATED_C_ABI_VERSION}u);
static_assert(BKNN_MANIFEST_SCHEMA_VERSION == {MANIFEST_SCHEMA_VERSION}u);
static_assert(BKNN_ARITHMETIC_PROFILE_VERSION == {ARITHMETIC_PROFILE_VERSION}u);
static_assert(BKNN_LAYOUT_NLC == 3u);
static_assert({macro}_C_ABI_VERSION == BKNN_C_ABI_VERSION);

int main() {{
    alignas({macro}_ARENA_ALIGNMENT)
        std::uint8_t arena[{macro}_ARENA_SIZE == 0u ? 1u : {macro}_ARENA_SIZE]{{}};
    std::int8_t input[{macro}_INPUT_SIZE]{{}};
    std::int8_t output[{macro}_OUTPUT_SIZE]{{}};
    {symbol}_infer({macro}_ARENA_SIZE == 0u ? nullptr : arena, input, output);
    return 0;
}}
''',
        encoding="utf-8",
    )
    executable = build / "runner"
    subprocess.run(
        [
            cxx,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-I",
            str(artifacts.output_dir),
            str(runner),
            *(str(path) for path in objects),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True)


def test_manifest_hashes_and_inventory_are_deterministic_and_verified(
    tmp_path: Path,
) -> None:
    first = bakenn.compile(tiny_cnn_graph(), tmp_path / "first")
    second = bakenn.compile(tiny_cnn_graph(), tmp_path / "second")
    first_manifest = load_manifest(first.artifacts.manifest)
    second_manifest = load_manifest(second.artifacts.manifest)

    assert first_manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert first_manifest["c_abi_version"] == GENERATED_C_ABI_VERSION
    assert (
        first_manifest["graph_fingerprints"]
        == second_manifest["graph_fingerprints"]
    )
    inventory = first_manifest["artifact_inventory"]
    paths = [record["path"] for record in inventory["files"]]
    assert paths == sorted(paths)
    assert first.artifacts.manifest.name not in paths
    assert inventory["manifest_included"] is False
    assert len(inventory["set_sha256"]) == 64
    assert len(first_manifest["manifest_payload_sha256"]) == 64
    assert first_manifest["compiler"]["package_version"] == bakenn.__version__

    tampered = first.artifacts.weights_source
    tampered.write_bytes(tampered.read_bytes() + b"\n")
    with pytest.raises(CompileError, match="(byte count|SHA-256) does not match"):
        load_manifest(first.artifacts.manifest)


def test_manifest_loader_rejects_duplicate_keys_and_unknown_schema(
    tmp_path: Path,
) -> None:
    compiled = bakenn.compile(tiny_cnn_graph(), tmp_path / "generated")
    manifest_text = compiled.artifacts.manifest.read_text(encoding="utf-8")
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        manifest_text.replace(
            f'"schema_version": {MANIFEST_SCHEMA_VERSION}',
            f'"schema_version": {MANIFEST_SCHEMA_VERSION}, '
            f'"schema_version": {MANIFEST_SCHEMA_VERSION}',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(CompileError, match="duplicate JSON key"):
        load_manifest(duplicate, verify_files=False)

    document = json.loads(manifest_text)
    document["schema_version"] = MANIFEST_SCHEMA_VERSION + 1
    with pytest.raises(CompileError, match="schema_version"):
        validate_manifest(document, verify_files=False)
    document = json.loads(manifest_text)
    document["unversioned_extension"] = True
    with pytest.raises(CompileError, match="unknown keys"):
        validate_manifest(document, verify_files=False)

    document = json.loads(manifest_text)
    document["operations"][0]["name"] = "tampered"
    with pytest.raises(CompileError, match="manifest_payload_sha256"):
        validate_manifest(document, verify_files=False)


def test_manifest_records_exact_measured_selection_provenance(tmp_path: Path) -> None:
    graph = linear_graph()
    plan = lower_to_plan(graph)
    workload = canonical_workload_key(plan, plan.steps[0])
    assert CORTEX_M4.toolchain is not None
    measurement = KernelCostMeasurement(
        kernel_id="optimized.linear_oi2.v1",
        workload=workload,
        cycles=123,
        toolchain=CORTEX_M4.toolchain,
        compiler_flags=CORTEX_M4.compiler_flags,
        evidence="physical-run.json",
    )
    target = replace(CORTEX_M4, measured_costs=(measurement,))
    compiled = bakenn.compile(
        graph,
        tmp_path / "measured",
        backend_options=bakenn.CBackendOptions(
            kernel_policy=bakenn.KernelPolicy.MEASURED,
            target=target,
        ),
    )
    selection = load_manifest(compiled.artifacts.manifest)["backend"]["selections"][0]
    assert selection["selection_basis"] == "measured_latency"
    assert selection["workload_key"] == workload
    assert selection["matched_measurement"] == measurement.manifest()


def test_source_identity_is_snapshotted_before_output_or_staging_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "repo_artifact"
    observed_calls = 0

    def clean_source_identity(version: str) -> dict[str, object]:
        nonlocal observed_calls
        observed_calls += 1
        assert version == bakenn.__version__
        assert not output.exists()
        assert _transaction_paths(tmp_path, output.name) == []
        return {
            "package_version": version,
            "build_identity": f"{version}+source.clean",
            "source_revision": "a" * 40,
            "source_dirty": False,
            "source_origin": "git",
        }

    monkeypatch.setattr(generator, "compiler_source_identity", clean_source_identity)
    compiled = bakenn.compile(tiny_cnn_graph(), output)

    assert observed_calls == 1
    assert load_manifest(compiled.artifacts.manifest)["compiler"] == {
        "package_version": bakenn.__version__,
        "build_identity": f"{bakenn.__version__}+source.clean",
        "source_revision": "a" * 40,
        "source_dirty": False,
        "source_origin": "git",
    }


def test_nonempty_unmanaged_output_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "generated"
    output.mkdir()
    sentinel = output / "user-owned.txt"
    sentinel.write_text("keep me", encoding="utf-8")

    with pytest.raises(CompileError, match="unmanaged output"):
        bakenn.compile(tiny_cnn_graph(), output)

    assert sentinel.read_text(encoding="utf-8") == "keep me"
    assert _transaction_paths(tmp_path, output.name) == []


def test_generation_failure_preserves_previous_artifact_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "generated"
    bakenn.compile(tiny_cnn_graph(), output)
    before = _tree_bytes(output)

    def fail_inventory(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        raise CompileError("injected inventory failure")

    monkeypatch.setattr(generator, "build_artifact_inventory", fail_inventory)
    with pytest.raises(CompileError, match="injected inventory failure"):
        bakenn.compile(tiny_cnn_graph(), output)

    assert _tree_bytes(output) == before
    assert load_manifest(next(output.glob("bknn_*_manifest.json")))
    assert _transaction_paths(tmp_path, output.name) == []


def test_successful_replacement_removes_stale_vendor_closure(tmp_path: Path) -> None:
    output = tmp_path / "generated"
    graph = cmsis_mlp_graph((32, 16, 4))
    cmsis_options = bakenn.CBackendOptions(
        kernel_policy=bakenn.KernelPolicy.STATIC_PRIORITY,
        enable_cmsis_nn=True,
        target=CORTEX_M4,
    )
    first = bakenn.compile(
        graph,
        output,
        backend_options=cmsis_options,
        target=CORTEX_M4,
    )
    assert first.artifacts.support_sources
    assert (output / "third_party").is_dir()

    second = bakenn.compile(graph, output, model_name="portable_replacement")
    assert second.artifacts.support_sources == ()
    assert not (output / "third_party").exists()
    assert not any(path.name.startswith("bknn_cmsis_mlp") for path in output.iterdir())
    assert load_manifest(second.artifacts.manifest)
    assert _transaction_paths(tmp_path, output.name) == []


def test_committed_output_survives_partial_old_backup_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "generated"
    first = bakenn.compile(tiny_cnn_graph(), output, model_name="old_model")
    assert load_manifest(first.artifacts.manifest)["model"] == "bknn_old_model"

    real_rmtree = shutil.rmtree

    def partial_failure(path: object, *args: object, **kwargs: object) -> None:
        candidate = Path(path)  # type: ignore[arg-type]
        if ".bakenn-backup-" in candidate.name:
            manifest = next(candidate.glob("bknn_*_manifest.json"))
            manifest.unlink()
            raise OSError("injected partial backup cleanup failure")
        real_rmtree(candidate, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(generator.shutil, "rmtree", partial_failure)
    second = bakenn.compile(tiny_cnn_graph(), output, model_name="new_model")

    manifest = load_manifest(second.artifacts.manifest)
    assert manifest["model"] == "bknn_new_model"
    assert second.artifacts.output_dir == output
    assert not any(path.name.startswith("bknn_old_model") for path in output.iterdir())
