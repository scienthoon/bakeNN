#!/usr/bin/env python3
"""Generate a tiny deterministic target-build fixture without PyTorch."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import bakenn


def smoke_graph() -> object:
    first_weight = (
        ((np.arange(16 * 32, dtype=np.float32) % 17.0) - 8.0) / 16.0
    ).reshape(16, 32)
    second_weight = (
        ((np.arange(4 * 16, dtype=np.float32) % 11.0) - 5.0) / 12.0
    ).reshape(4, 16)
    model = bakenn.FloatMLP(
        (
            bakenn.FloatLinear(
                first_weight,
                np.linspace(-0.125, 0.125, 16, dtype=np.float32),
                True,
                "hidden",
            ),
            bakenn.FloatLinear(
                second_weight,
                np.asarray([0.0, 0.125, -0.125, 0.0625], dtype=np.float32),
                False,
                "output",
            ),
        ),
        "target_smoke",
    )
    calibration = np.stack(
        (
            np.linspace(-1.0, 1.0, 32, dtype=np.float32),
            np.linspace(1.0, -1.0, 32, dtype=np.float32),
            np.sin(np.arange(32, dtype=np.float32)),
        )
    )
    return bakenn.quantize_ptq(model, calibration)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=sorted(bakenn.TARGET_PROFILES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--kernel-policy",
        choices=("portable", "auto", "require_optimized"),
        default="auto",
    )
    parser.add_argument("--cross-build", action="store_true")
    parser.add_argument("--esp-idf", action="store_true")
    parser.add_argument(
        "--zephyr-board",
        choices=("nrf52840dk_nrf52840", "nrf52dk_nrf52832", "disco_l475_iot1"),
    )
    arguments = parser.parse_args()
    descriptor = bakenn.resolve_target(arguments.target)
    generated = arguments.output / "generated"
    compiled = bakenn.compile(
        smoke_graph(),
        generated,
        model_name="target_smoke",
        target=descriptor,
        backend_options=bakenn.CBackendOptions(
            kernel_policy=bakenn.KernelPolicy(arguments.kernel_policy),
            target=descriptor,
        ),
    )
    print(compiled.artifacts.manifest)
    if arguments.cross_build:
        report = bakenn.build_freestanding_elf(
            compiled.artifacts, descriptor, arguments.output / "cross"
        )
        print(report.write_json(arguments.output / "cross" / "report.json"))
    if arguments.esp_idf:
        project = bakenn.export_esp_idf_project(
            compiled.artifacts, descriptor, arguments.output / "esp_idf"
        )
        print(project.root)
    if arguments.zephyr_board:
        project = bakenn.export_zephyr_project(
            compiled.artifacts,
            descriptor,
            arguments.output / "zephyr",
            board=arguments.zephyr_board,
        )
        print(project.root)


if __name__ == "__main__":
    main()
