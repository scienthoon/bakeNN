#!/usr/bin/env python3
"""Dependency-free validator for BakeNN/TFLM comparison result documents."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any


class ValidationError(ValueError):
    pass


def _object(value: Any, path: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{path} must be an object")
    unknown = set(value) - fields
    if unknown:
        raise ValidationError(f"{path} has unknown fields: {sorted(unknown)}")
    return value


def _required(mapping: dict[str, Any], fields: set[str], path: str) -> None:
    missing = fields - set(mapping)
    if missing:
        raise ValidationError(f"{path} is missing fields: {sorted(missing)}")


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{path} must be a non-empty string")
    return value


def _integer(value: Any, path: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValidationError(f"{path} must be an integer >= {minimum}")
    return value


def _number(value: Any, path: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{path} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < minimum:
        raise ValidationError(f"{path} must be finite and >= {minimum}")
    return normalized


def _measurement(value: Any, path: str, metric: str) -> None:
    item = _object(value, path, {"status", "value", "reason"})
    _required(item, {"status", "value", "reason"}, path)
    status = item["status"]
    if status not in ("measured", "unmeasured"):
        raise ValidationError(f"{path}.status must be measured or unmeasured")
    if status == "unmeasured":
        if item["value"] is not None:
            raise ValidationError(f"{path}.value must be null while unmeasured")
        _text(item["reason"], f"{path}.reason")
        return
    if item["reason"] is not None:
        raise ValidationError(f"{path}.reason must be null while measured")
    if metric in {"output_max_abs_error_lsb"}:
        _number(item["value"], f"{path}.value")
    else:
        _integer(item["value"], f"{path}.value")


_METRICS = (
    "elf_flash_text_bytes",
    "elf_flash_rodata_bytes",
    "elf_flash_data_load_bytes",
    "elf_flash_total_bytes",
    "peak_sram_arena_bytes",
    "peak_sram_runtime_metadata_bytes",
    "peak_sram_stack_bytes",
    "peak_sram_static_data_bytes",
    "peak_sram_total_bytes",
    "init_cycles",
    "inference_median_cycles",
    "inference_p95_cycles",
    "output_compared_bytes",
    "output_mismatched_bytes",
    "output_max_abs_error_lsb",
)


def _implementation(value: Any, path: str) -> None:
    item = _object(value, path, {"version", "source_revision", "metrics"})
    _required(item, {"version", "source_revision", "metrics"}, path)
    _text(item["version"], f"{path}.version")
    _text(item["source_revision"], f"{path}.source_revision")
    metrics = _object(item["metrics"], f"{path}.metrics", set(_METRICS))
    _required(metrics, set(_METRICS), f"{path}.metrics")
    for metric in _METRICS:
        _measurement(metrics[metric], f"{path}.metrics.{metric}", metric)
    measured = {
        name: value["value"]
        for name, value in metrics.items()
        if value["status"] == "measured"
    }
    flash_parts = (
        "elf_flash_text_bytes",
        "elf_flash_rodata_bytes",
        "elf_flash_data_load_bytes",
    )
    if all(name in measured for name in (*flash_parts, "elf_flash_total_bytes")):
        if sum(measured[name] for name in flash_parts) != measured["elf_flash_total_bytes"]:
            raise ValidationError(f"{path}: measured ELF flash total must equal its three sections")
    sram_parts = (
        "peak_sram_arena_bytes",
        "peak_sram_runtime_metadata_bytes",
        "peak_sram_stack_bytes",
        "peak_sram_static_data_bytes",
    )
    if all(name in measured for name in (*sram_parts, "peak_sram_total_bytes")):
        if sum(measured[name] for name in sram_parts) != measured["peak_sram_total_bytes"]:
            raise ValidationError(f"{path}: measured peak SRAM total must equal its components")
    if all(name in measured for name in ("inference_median_cycles", "inference_p95_cycles")):
        if measured["inference_p95_cycles"] < measured["inference_median_cycles"]:
            raise ValidationError(f"{path}: inference p95 cycles cannot be below median")
    if all(name in measured for name in ("output_compared_bytes", "output_mismatched_bytes")):
        if measured["output_mismatched_bytes"] > measured["output_compared_bytes"]:
            raise ValidationError(f"{path}: mismatched output bytes exceed compared bytes")


def validate_result(document: Any) -> dict[str, Any]:
    root = _object(
        document,
        "$",
        {
            "schema_version",
            "benchmark_id",
            "timestamp_utc",
            "model",
            "target",
            "compiler",
            "protocol",
            "implementations",
        },
    )
    _required(
        root,
        {
            "schema_version",
            "benchmark_id",
            "timestamp_utc",
            "model",
            "target",
            "compiler",
            "protocol",
            "implementations",
        },
        "$",
    )
    if root["schema_version"] != 1:
        raise ValidationError("$.schema_version must equal 1")
    for field in ("benchmark_id", "timestamp_utc"):
        _text(root[field], f"$.{field}")

    model = _object(root["model"], "$.model", {"name", "artifact_sha256", "input_sha256"})
    _required(model, {"name", "artifact_sha256", "input_sha256"}, "$.model")
    for field in model:
        _text(model[field], f"$.model.{field}")

    target = _object(
        root["target"], "$.target", {"board", "mcu", "architecture", "clock_hz"}
    )
    _required(target, {"board", "mcu", "architecture", "clock_hz"}, "$.target")
    for field in ("board", "mcu", "architecture"):
        _text(target[field], f"$.target.{field}")
    _integer(target["clock_hz"], "$.target.clock_hz", 1)

    compiler = _object(root["compiler"], "$.compiler", {"name", "version", "flags", "linker_script"})
    _required(compiler, {"name", "version", "flags", "linker_script"}, "$.compiler")
    for field in ("name", "version", "linker_script"):
        _text(compiler[field], f"$.compiler.{field}")
    if not isinstance(compiler["flags"], list) or not all(
        isinstance(flag, str) and flag for flag in compiler["flags"]
    ):
        raise ValidationError("$.compiler.flags must be an array of non-empty strings")

    protocol = _object(
        root["protocol"],
        "$.protocol",
        {"warmup_runs", "measured_runs", "cycle_counter", "interrupt_policy", "input_count"},
    )
    _required(
        protocol,
        {"warmup_runs", "measured_runs", "cycle_counter", "interrupt_policy", "input_count"},
        "$.protocol",
    )
    _integer(protocol["warmup_runs"], "$.protocol.warmup_runs")
    _integer(protocol["measured_runs"], "$.protocol.measured_runs", 1)
    _integer(protocol["input_count"], "$.protocol.input_count", 1)
    _text(protocol["cycle_counter"], "$.protocol.cycle_counter")
    _text(protocol["interrupt_policy"], "$.protocol.interrupt_policy")

    implementations = _object(root["implementations"], "$.implementations", {"bakenn", "tflm"})
    _required(implementations, {"bakenn", "tflm"}, "$.implementations")
    _implementation(implementations["bakenn"], "$.implementations.bakenn")
    _implementation(implementations["tflm"], "$.implementations.tflm")
    return root


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("usage: validate_result.py RESULT.json", file=sys.stderr)
        return 2
    path = Path(arguments[0])
    try:
        validate_result(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        print(f"invalid benchmark result: {error}", file=sys.stderr)
        return 1
    print(f"valid benchmark result: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ValidationError", "validate_result"]
