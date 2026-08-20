from __future__ import annotations

import hashlib
import json
import os
import re
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
    size_path = root / "benchmarks/esp32/results/mnist_trained_esp32_size.txt"
    size_output = size_path.read_text(encoding="utf-8")

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
    assert artifacts["size_output_sha256"] == hashlib.sha256(
        size_path.read_bytes()
    ).hexdigest()

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

    raw_cycle_lines = [
        line for line in uart.splitlines() if line.startswith("BAKENN_MNIST_CYCLES=")
    ]
    assert len(raw_cycle_lines) == 1
    raw_cycles = [
        int(value)
        for value in raw_cycle_lines[0].removeprefix(
            "BAKENN_MNIST_CYCLES="
        ).split(",")
    ]
    assert len(raw_cycles) == result["latency"]["measured_runs"] == 101
    sorted_cycles = sorted(raw_cycles)
    assert sorted_cycles[50] == result["latency"]["median_cycles"]
    assert sorted_cycles[95] == result["latency"]["p95_cycles"]

    summary_line = next(
        line for line in uart.splitlines() if line.startswith("BAKENN_MNIST target=")
    )
    summary = dict(field.split("=", 1) for field in summary_line.split()[1:])
    assert int(summary["first_cycles"]) == result["latency"]["first_cycles"]
    assert int(summary["median_cycles"]) == sorted_cycles[50]
    assert int(summary["p95_cycles"]) == sorted_cycles[95]

    assert "$ idf.py size" in size_output
    assert "$ idf.py size-components" in size_output

    def size_table_value(name: str) -> int:
        match = re.search(
            rf"^│\s*{re.escape(name)}\s*│\s*(\d+)\s*│",
            size_output,
            re.MULTILINE,
        )
        assert match is not None
        return int(match.group(1))

    assert size_table_value("Flash Data") == memory["flash_data_bytes"]
    assert size_table_value("Flash Code") == memory["flash_code_bytes"]
    assert size_table_value("IRAM") == memory["iram_bytes"]
    assert size_table_value("DRAM") == memory["dram_bytes"]
    assert size_table_value("RTC SLOW") == memory["rtc_slow_bytes"]

    app_binary = re.search(
        r"bakenn_target_smoke\.bin binary size 0x([0-9a-f]+) bytes",
        size_output,
    )
    total_image = re.search(r"Total image size: (\d+) bytes", size_output)
    assert app_binary is not None
    assert total_image is not None
    assert int(app_binary.group(1), 16) == memory["app_binary_bytes"]
    assert int(total_image.group(1)) == memory["elf_total_image_bytes"]

    model_archive = re.search(
        r"^libbakenn_model\.a,0,0,0,0,0,(\d+),(\d+),0$",
        size_output,
        re.MULTILINE,
    )
    assert model_archive is not None
    model_code = int(model_archive.group(1))
    model_data = int(model_archive.group(2))
    assert model_code == memory["model_component_flash_code_bytes"]
    assert model_data == memory["model_component_flash_data_bytes"]
    assert model_code + model_data == memory["model_component_flash_bytes"]
