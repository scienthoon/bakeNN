#!/usr/bin/env python3
"""Generate a tiny deterministic target-build fixture without PyTorch."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import bakenn


def smoke_graph() -> object:
    model = bakenn.FloatMLP(
        (
            bakenn.FloatLinear(
                np.asarray(
                    [
                        [1.0, -0.5, 0.25, 0.0],
                        [-0.75, 0.25, 0.5, 1.0],
                        [0.125, 0.5, -1.0, 0.25],
                    ],
                    dtype=np.float32,
                ),
                np.asarray([0.125, -0.25, 0.0625], dtype=np.float32),
                True,
                "hidden",
            ),
            bakenn.FloatLinear(
                np.asarray([[0.5, -1.0, 0.25], [-0.25, 0.75, 1.0]], dtype=np.float32),
                np.asarray([0.0, 0.125], dtype=np.float32),
                False,
                "output",
            ),
        ),
        "target_smoke",
    )
    calibration = np.asarray(
        [[-1.0, 0.0, 0.5, 1.0], [1.0, -0.5, 0.0, -1.0]], dtype=np.float32
    )
    return bakenn.quantize_ptq(model, calibration)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=sorted(bakenn.TARGET_PROFILES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cross-build", action="store_true")
    parser.add_argument("--esp-idf", action="store_true")
    arguments = parser.parse_args()
    descriptor = bakenn.resolve_target(arguments.target)
    generated = arguments.output / "generated"
    compiled = bakenn.compile(
        smoke_graph(), generated, model_name="target_smoke", target=descriptor
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


if __name__ == "__main__":
    main()
