from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from scripts.verify_mnist_evidence import verify


def test_frozen_mnist_evidence_compiles_and_matches_expected_bytes() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence_dir = root / "examples/mnist/evidence"
    evidence = json.loads(
        (evidence_dir / "mnist_evidence.json").read_text(encoding="utf-8")
    )
    result = verify(evidence_dir, os.environ.get("CC", "cc"))

    assert evidence["training"]["performed"] is True
    assert evidence["training"]["epochs"] == 4
    assert evidence["working_tree_dirty"] is False
    assert len(evidence["source_commit"]) == 40
    assert result["result"] == "PASS"
    assert result["physical_measurement"] is False
    assert result["verified_payload_files"] >= 17
    assert result["compared_output_bytes"] == 1000
    assert result["mismatched_output_bytes"] == 0
    assert result["output_fnv1a"] == "0x55fb9e60"
    assert result["output_sha256"] == (
        "cdbbfc38bf5f17a2867b17a8a28e643043e50733063e246b0bc85cbeaae30ba4"
    )


def test_physical_esp32_result_matches_frozen_mnist_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence = json.loads(
        (root / "examples/mnist/evidence/mnist_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    cross_build = json.loads(
        (root / "benchmarks/cross_build/results/mnist_esp32_cross_build.json").read_text(
            encoding="utf-8"
        )
    )
    result = json.loads(
        (root / "benchmarks/esp32/results/mnist_trained_esp32.json").read_text(
            encoding="utf-8"
        )
    )
    uart_path = root / "benchmarks/esp32/results/mnist_trained_esp32_uart.txt"
    uart = uart_path.read_text(encoding="utf-8")

    assert result["evidence_class"] == "physical_board"
    assert result["source"]["submission_commit"] == (
        "23f3a4f744135ad02bc738a77aee531f7ff2a751"
    )
    assert result["correctness"] == {
        "samples": 100,
        "correct": 99,
        "accuracy": 0.99,
        "compared_output_bytes": 1000,
        "mismatched_output_bytes": 0,
        "output_fnv1a_32": "0x55fb9e60",
        "expected_and_board_output_sha256": (
            "cdbbfc38bf5f17a2867b17a8a28e643043e50733063e246b0bc85cbeaae30ba4"
        ),
        "status": "pass",
    }

    artifacts = result["artifacts"]
    assert artifacts["checkpoint_file_sha256"] == evidence["checkpoint"]["file_sha256"]
    assert artifacts["checkpoint_logical_sha256"] == (
        evidence["checkpoint"]["logical_tensor_sha256"]
    )
    assert artifacts["calibration_corpus_sha256"] == (
        evidence["calibration"]["corpus_sha256"]
    )
    assert artifacts["physical_input_sha256"] == (
        evidence["physical_test_corpus"]["quantized_input_sha256"]
    )
    assert artifacts["expected_output_sha256"] == (
        evidence["physical_test_corpus"]["expected_output_sha256"]
    )
    assert artifacts["frozen_portable_generated_set_sha256"] == (
        evidence["generated_artifacts"]["set_sha256"]
    )
    assert artifacts["benchmark_contract_sha256"] == (
        cross_build["artifacts"]["contract_sha256"]
    )
    assert artifacts["uart_sha256"] == hashlib.sha256(uart_path.read_bytes()).hexdigest()

    for token in (
        "samples=100",
        "correct=99",
        "compared_bytes=1000",
        "mismatches=0",
        "BAKENN_MNIST_OUTPUT_FNV1A=0x55fb9e60",
        f"checkpoint={artifacts['checkpoint_logical_sha256']}",
        f"calibration={artifacts['calibration_corpus_sha256']}",
        f"input={artifacts['physical_input_sha256']}",
        f"expected={artifacts['expected_output_sha256']}",
    ):
        assert token in uart

    memory = result["memory"]
    assert memory["static_sram_iram_plus_dram_bytes"] == (
        memory["iram_bytes"] + memory["dram_bytes"]
    )
    assert memory["stack_high_water_free_bytes"] > 0
