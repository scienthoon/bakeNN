#!/usr/bin/env python3
"""Compile a deterministic FP32 PyTorch CNN into an ESP32-S3 project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import bakenn
from bakenn.targets import ESP32_S3


class DemoCNN(torch.nn.Module):
    """Small RGB classifier covering Conv, Depthwise, 1x1, pool and FC."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.depthwise = torch.nn.Conv2d(
            16, 16, 3, padding=1, groups=16
        )
        self.pointwise = torch.nn.Conv2d(16, 16, 1)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(16, 16)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.relu(self.stem(value))
        value = torch.relu(self.depthwise(value))
        value = torch.relu(self.pointwise(value))
        value = self.pool(value)
        value = torch.flatten(value, 1)
        return self.classifier(value)


def _calibration_data() -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260816)
    random = torch.randn(12, 3, 16, 16, generator=generator)
    gradient = torch.linspace(-2.0, 2.0, 16, dtype=torch.float32)
    random[0] = gradient.view(1, 1, 16).expand(3, 16, 16)
    random[1] = gradient.view(1, 16, 1).expand(3, 16, 16)
    return random


def generate(output: Path) -> dict[str, object]:
    torch.manual_seed(20260816)
    model = DemoCNN().eval()
    calibration = _calibration_data()
    generated = output / "generated"

    compiled = bakenn.compile_torch_ptq(
        model,
        calibration[:1],
        calibration,
        generated,
        name="esp32s3_demo_cnn",
        target=ESP32_S3,
        backend_options=bakenn.CBackendOptions(
            kernel_policy=bakenn.KernelPolicy.AUTO,
            enable_esp_nn=True,
            target=ESP32_S3,
        ),
    )
    project = bakenn.export_esp_idf_project(
        compiled.artifacts,
        ESP32_S3,
        output / "esp_idf",
    )

    selections = [
        {
            "step": item.step_name,
            "kernel": item.kernel_id,
            "optimized": item.optimized,
            "reason": item.reason,
        }
        for item in compiled.artifacts.backend_plan.selections
    ]
    selected_ids = {str(item["kernel"]) for item in selections}
    required_fragments = (
        ".conv2d_s8.",
        ".depthwise_conv2d_s8.",
        ".linear_per_channel_s8.",
    )
    missing = [
        fragment
        for fragment in required_fragments
        if not any(
            kernel.startswith("esp_nn.esp32s3.") and fragment in kernel
            for kernel in selected_ids
        )
    ]
    if missing:
        raise RuntimeError(f"demo did not select required ESP-NN paths: {missing}")

    manifest = json.loads(
        compiled.artifacts.manifest.read_text(encoding="utf-8")
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "model": "deterministic untrained FP32 DemoCNN",
        "input": [1, 3, 16, 16],
        "calibration_samples": int(calibration.shape[0]),
        "target": "esp32s3",
        "generated_manifest": str(compiled.artifacts.manifest),
        "esp_idf_project": str(project.root),
        "arena_bytes": manifest["arena_bytes"],
        "constant_bytes": manifest["constant_bytes"],
        "selections": selections,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "demo_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/esp32s3_end_to_end"),
    )
    arguments = parser.parse_args()
    summary = generate(arguments.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nBuild and flash:")
    print(f"  cd {summary['esp_idf_project']}")
    print("  idf.py set-target esp32s3")
    print("  idf.py build")
    print("  idf.py -p PORT flash monitor")


if __name__ == "__main__":
    main()
