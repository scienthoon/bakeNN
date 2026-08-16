from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_microtvm_cross_build_evidence_is_complete_and_nonphysical() -> None:
    root = Path(__file__).resolve().parents[1]
    results = root / "benchmarks/microtvm_compare/results"
    record = json.loads((results / "mnist_cortex_m4_cross_build.json").read_text())

    assert record["evidence_class"] == "boardless_cross_build"
    assert record["performance_claim_permitted"] is False
    assert record["contract"]["samples"] == 100
    assert record["contract"]["operator_counts"] == {
        "CONV_2D": 2,
        "FULLY_CONNECTED": 1,
        "MAX_POOL_2D": 2,
        "RESHAPE": 1,
    }
    expected_differential = {
        "compared_bytes": 1000,
        "max_abs_lsb_error": 0,
        "mismatched_bytes": 0,
    }
    assert record["host_differential"]["bakenn_vs_reference"] == expected_differential
    assert record["host_differential"]["microtvm_vs_reference"] == expected_differential
    assert record["host_differential"]["microtvm_vs_bakenn_mismatched_bytes"] == 0
    assert record["microtvm"]["workspace_size_bytes"] == 4064
    assert record["runtime"]["physical_cycles"]["status"] == "unmeasured"

    existing_input = root / "examples/mnist/evidence/physical_test_inputs_int8.bin"
    assert _sha256(existing_input) == record["contract"]["input_sha256"]
    expected = results / "mnist_common_expected_int8.bin"
    assert _sha256(expected) == record["contract"]["expected_output_sha256"]

    tflite = results / "mnist_common_int8.tflite"
    assert tflite.read_bytes()[4:8] == b"TFL3"
    assert _sha256(tflite) == record["artifacts"]["tflite_sha256"]

    for item in record["artifacts"]["files"]:
        path = results / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert _sha256(path) == item["sha256"]

    generated = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((results / "microtvm_codegen").glob("*.c"))
    )
    for symbol in record["microtvm"]["cmsis_nn_calls"]:
        assert symbol in generated

    linked = record["cortex_m4_cross_link"]
    assert linked["bakenn"]["flash_load_bytes"] < linked["microtvm"]["flash_load_bytes"]
    assert linked["bakenn"]["arena_bytes"] == linked["microtvm"]["usmp_workspace_bytes"]
