#!/usr/bin/env python3
"""Generate a tiny deterministic target-build fixture without PyTorch."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import bakenn
from bakenn.ir import (
    Conv2DOp,
    DType,
    DepthwiseConv2DOp,
    Layout,
    PerAxisQParams,
    PerTensorQParams,
    QuantizedGraph,
    TensorType,
)


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


def esp_nn_smoke_graph() -> QuantizedGraph:
    """Small Conv+Depthwise graph that forces both ESP-NN spatial paths."""

    input_q = PerTensorQParams(0.25, -7)
    hidden_q = PerTensorQParams(0.5, 3)
    output_q = PerTensorQParams(0.75, -2)
    conv_scales = (0.125, 0.25, 0.375, 0.5)
    depthwise_scales = (0.25, 0.375, 0.5, 0.625)
    conv_weight = (
        (np.arange(16, dtype=np.int16) * 5 + 3) % 17 - 8
    ).reshape(4, 1, 1, 4).astype(np.int8)
    depthwise_weight = (
        (np.arange(36, dtype=np.int16) * 7 + 1) % 19 - 9
    ).reshape(3, 3, 4).astype(np.int8)
    conv_bias = np.asarray((3, -5, 7, -11), dtype=np.int32)
    depthwise_bias = np.asarray((-13, 17, -19, 23), dtype=np.int32)
    return QuantizedGraph(
        name="target_esp_nn_smoke",
        values={
            "input": TensorType((1, 4, 4, 4), DType.INT8, Layout.NHWC, input_q),
            "conv_weight": TensorType(
                (4, 1, 1, 4),
                DType.INT8,
                Layout.OHWI,
                PerAxisQParams(conv_scales, (0,) * 4, 0),
            ),
            "conv_bias": TensorType(
                (4,),
                DType.INT32,
                Layout.C,
                PerAxisQParams(
                    tuple(input_q.scale * scale for scale in conv_scales),
                    (0,) * 4,
                    0,
                ),
            ),
            "hidden": TensorType((1, 4, 4, 4), DType.INT8, Layout.NHWC, hidden_q),
            "depthwise_weight": TensorType(
                (3, 3, 4),
                DType.INT8,
                Layout.HWO,
                PerAxisQParams(depthwise_scales, (0,) * 4, 2),
            ),
            "depthwise_bias": TensorType(
                (4,),
                DType.INT32,
                Layout.C,
                PerAxisQParams(
                    tuple(hidden_q.scale * scale for scale in depthwise_scales),
                    (0,) * 4,
                    0,
                ),
            ),
            "output": TensorType((1, 4, 4, 4), DType.INT8, Layout.NHWC, output_q),
        },
        constants={
            "conv_weight": conv_weight,
            "conv_bias": conv_bias,
            "depthwise_weight": depthwise_weight,
            "depthwise_bias": depthwise_bias,
        },
        ops=(
            Conv2DOp(
                "conv",
                "input",
                "conv_weight",
                "conv_bias",
                "hidden",
                activation_min=-96,
                activation_max=111,
            ),
            DepthwiseConv2DOp(
                "depthwise",
                "hidden",
                "depthwise_weight",
                "depthwise_bias",
                "output",
                padding=(1, 1, 1, 1),
                activation_min=-101,
                activation_max=103,
            ),
        ),
        inputs=("input",),
        outputs=("output",),
    )


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
        "--esp-nn",
        action="store_true",
        help="select pinned ESP-NN kernels and use a Conv+Depthwise smoke graph",
    )
    parser.add_argument(
        "--zephyr-board",
        choices=("nrf52840dk_nrf52840", "nrf52dk_nrf52832", "disco_l475_iot1"),
    )
    arguments = parser.parse_args()
    descriptor = bakenn.resolve_target(arguments.target)
    generated = arguments.output / "generated"
    compiled = bakenn.compile(
        esp_nn_smoke_graph() if arguments.esp_nn else smoke_graph(),
        generated,
        model_name="target_smoke",
        target=descriptor,
        backend_options=bakenn.CBackendOptions(
            kernel_policy=bakenn.KernelPolicy(arguments.kernel_policy),
            enable_esp_nn=arguments.esp_nn,
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
