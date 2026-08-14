from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks" / "tflm_compare"
VALIDATOR = BENCHMARK / "validate_result.py"
EXAMPLE = BENCHMARK / "example_result.json"


def _validator_module():  # type: ignore[no-untyped-def]
    specification = importlib.util.spec_from_file_location("bakenn_benchmark_validator", VALIDATOR)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _example() -> dict[str, object]:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def _measured(value: int | float) -> dict[str, object]:
    return {"status": "measured", "value": value, "reason": None}


def test_unmeasured_example_is_valid_and_contains_no_claimed_metric() -> None:
    module = _validator_module()
    document = _example()
    assert module.validate_result(document) is document
    for implementation in document["implementations"].values():  # type: ignore[union-attr]
        for metric in implementation["metrics"].values():
            assert metric["status"] == "unmeasured"
            assert metric["value"] is None
            assert isinstance(metric["reason"], str) and metric["reason"]


def test_dependency_free_cli_accepts_example() -> None:
    completed = subprocess.run(
        [sys.executable, str(VALIDATOR), str(EXAMPLE)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "valid benchmark result" in completed.stdout


def test_validator_accepts_consistent_measured_metrics() -> None:
    module = _validator_module()
    document = _example()
    for implementation in document["implementations"].values():  # type: ignore[union-attr]
        metrics = implementation["metrics"]
        metrics.update(
            {
                "elf_flash_text_bytes": _measured(100),
                "elf_flash_rodata_bytes": _measured(40),
                "elf_flash_data_load_bytes": _measured(8),
                "elf_flash_total_bytes": _measured(148),
                "peak_sram_arena_bytes": _measured(200),
                "peak_sram_runtime_metadata_bytes": _measured(20),
                "peak_sram_stack_bytes": _measured(30),
                "peak_sram_static_data_bytes": _measured(10),
                "peak_sram_total_bytes": _measured(260),
                "init_cycles": _measured(5),
                "inference_median_cycles": _measured(1000),
                "inference_p95_cycles": _measured(1100),
                "output_compared_bytes": _measured(64),
                "output_mismatched_bytes": _measured(0),
                "output_max_abs_error_lsb": _measured(0.0),
            }
        )
    module.validate_result(document)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["implementations"]["bakenn"]["metrics"][
                "init_cycles"
            ].update({"status": "measured", "value": None, "reason": None}),
            "must be an integer",
        ),
        (
            lambda value: value["implementations"]["bakenn"]["metrics"][
                "init_cycles"
            ].update({"status": "unmeasured", "value": 7}),
            "must be null",
        ),
        (
            lambda value: value["implementations"]["bakenn"]["metrics"].pop(
                "peak_sram_total_bytes"
            ),
            "missing fields",
        ),
        (
            lambda value: value["implementations"].update({"third_runtime": {}}),
            "unknown fields",
        ),
    ],
)
def test_validator_rejects_malformed_results(mutation, message: str) -> None:  # type: ignore[no-untyped-def]
    module = _validator_module()
    document = deepcopy(_example())
    mutation(document)
    with pytest.raises(module.ValidationError, match=message):
        module.validate_result(document)


def test_validator_rejects_inconsistent_measured_totals_and_ordering() -> None:
    module = _validator_module()
    base = _example()
    metrics = base["implementations"]["bakenn"]["metrics"]  # type: ignore[index]
    metrics["elf_flash_text_bytes"] = _measured(10)
    metrics["elf_flash_rodata_bytes"] = _measured(20)
    metrics["elf_flash_data_load_bytes"] = _measured(3)
    metrics["elf_flash_total_bytes"] = _measured(32)
    with pytest.raises(module.ValidationError, match="ELF flash total"):
        module.validate_result(base)

    latency = _example()
    latency_metrics = latency["implementations"]["tflm"]["metrics"]  # type: ignore[index]
    latency_metrics["inference_median_cycles"] = _measured(101)
    latency_metrics["inference_p95_cycles"] = _measured(100)
    with pytest.raises(module.ValidationError, match="p95 cycles"):
        module.validate_result(latency)

    output = _example()
    output_metrics = output["implementations"]["bakenn"]["metrics"]  # type: ignore[index]
    output_metrics["output_compared_bytes"] = _measured(8)
    output_metrics["output_mismatched_bytes"] = _measured(9)
    with pytest.raises(module.ValidationError, match="mismatched output bytes"):
        module.validate_result(output)
