#!/usr/bin/env python3
"""Host smoke benchmark for the P2 Linear/Conv/Depthwise specializations.

This is useful for catching catastrophic regressions in code shape. It is not
an MCU result and must not be used for BakeNN-versus-TFLM claims.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile

import numpy as np

import bakenn
from bakenn.ir import (
    DType,
    Conv2DOp,
    DepthwiseConv2DOp,
    Layout,
    LinearOp,
    PerAxisQParams,
    PerTensorQParams,
    QuantizedGraph,
    TensorType,
)


def _linear_graph() -> QuantizedGraph:
    input_count = 392
    output_count = 10
    input_q = PerTensorQParams(0.03125, -3)
    output_q = PerTensorQParams(0.0625, 1)
    weight_scales = tuple(0.0078125 for _ in range(output_count))
    rng = np.random.default_rng(20260814)
    weight = rng.integers(-127, 128, (output_count, input_count), dtype=np.int16).astype(
        np.int8
    )
    bias = rng.integers(-1000, 1001, output_count, dtype=np.int32)
    return QuantizedGraph(
        name="host_linear_392x10",
        values={
            "input": TensorType((1, input_count), DType.INT8, Layout.NC, input_q),
            "weight": TensorType(
                (output_count, input_count),
                DType.INT8,
                Layout.OI,
                PerAxisQParams(weight_scales, (0,) * output_count, 0),
            ),
            "bias": TensorType(
                (output_count,),
                DType.INT32,
                Layout.C,
                PerAxisQParams(
                    tuple(input_q.scale * scale for scale in weight_scales),
                    (0,) * output_count,
                    0,
                ),
            ),
            "output": TensorType((1, output_count), DType.INT8, Layout.NC, output_q),
        },
        constants={"weight": weight, "bias": bias},
        ops=(LinearOp("linear", "input", "weight", "bias", "output"),),
        inputs=("input",),
        outputs=("output",),
    )


def _conv1x1_graph() -> QuantizedGraph:
    input_shape = (1, 16, 16, 32)
    output_channels = 64
    input_q = PerTensorQParams(0.03125, -3)
    output_q = PerTensorQParams(0.0625, 1)
    weight_scales = tuple(0.0078125 for _ in range(output_channels))
    rng = np.random.default_rng(20260815)
    weight = rng.integers(
        -127, 128, (output_channels, 1, 1, input_shape[3]), dtype=np.int16
    ).astype(np.int8)
    bias = rng.integers(-1000, 1001, output_channels, dtype=np.int32)
    return QuantizedGraph(
        name="host_conv1x1_16x16x32x64",
        values={
            "input": TensorType(input_shape, DType.INT8, Layout.NHWC, input_q),
            "weight": TensorType(
                weight.shape,
                DType.INT8,
                Layout.OHWI,
                PerAxisQParams(weight_scales, (0,) * output_channels, 0),
            ),
            "bias": TensorType(
                (output_channels,),
                DType.INT32,
                Layout.C,
                PerAxisQParams(
                    tuple(input_q.scale * scale for scale in weight_scales),
                    (0,) * output_channels,
                    0,
                ),
            ),
            "output": TensorType(
                (1, 16, 16, output_channels),
                DType.INT8,
                Layout.NHWC,
                output_q,
            ),
        },
        constants={"weight": weight, "bias": bias},
        ops=(Conv2DOp("conv1x1", "input", "weight", "bias", "output"),),
        inputs=("input",),
        outputs=("output",),
    )


def _depthwise_graph() -> QuantizedGraph:
    input_shape = (1, 32, 32, 32)
    channels = input_shape[3]
    input_q = PerTensorQParams(0.03125, 5)
    output_q = PerTensorQParams(0.0625, -1)
    weight_scales = tuple(0.0078125 for _ in range(channels))
    rng = np.random.default_rng(20260816)
    weight = rng.integers(-127, 128, (3, 3, channels), dtype=np.int16).astype(
        np.int8
    )
    bias = rng.integers(-1000, 1001, channels, dtype=np.int32)
    return QuantizedGraph(
        name="host_depthwise_32x32x32",
        values={
            "input": TensorType(input_shape, DType.INT8, Layout.NHWC, input_q),
            "weight": TensorType(
                weight.shape,
                DType.INT8,
                Layout.HWO,
                PerAxisQParams(weight_scales, (0,) * channels, 2),
            ),
            "bias": TensorType(
                (channels,),
                DType.INT32,
                Layout.C,
                PerAxisQParams(
                    tuple(input_q.scale * scale for scale in weight_scales),
                    (0,) * channels,
                    0,
                ),
            ),
            "output": TensorType(input_shape, DType.INT8, Layout.NHWC, output_q),
        },
        constants={"weight": weight, "bias": bias},
        ops=(
            DepthwiseConv2DOp(
                "depthwise",
                "input",
                "weight",
                "bias",
                "output",
                depth_multiplier=1,
                padding=(1, 1, 1, 1),
            ),
        ),
        inputs=("input",),
        outputs=("output",),
    )


_GRAPH_FACTORIES = {
    "linear": _linear_graph,
    "conv1x1": _conv1x1_graph,
    "depthwise3x3": _depthwise_graph,
}

_DEFAULT_ITERATIONS = {
    "linear": 200_000,
    "conv1x1": 200,
    "depthwise3x3": 1_000,
}


def _runner(
    portable: bakenn.compiler.CompiledModel,
    optimized: bakenn.compiler.CompiledModel,
) -> str:
    portable_manifest = json.loads(portable.artifacts.manifest.read_text(encoding="utf-8"))
    optimized_manifest = json.loads(optimized.artifacts.manifest.read_text(encoding="utf-8"))
    portable_symbol = portable_manifest["model"]
    optimized_symbol = optimized_manifest["model"]
    macro = portable_symbol.upper()
    return f"""#define _POSIX_C_SOURCE 200809L
#include \"{portable.artifacts.header.name}\"
#include \"{optimized.artifacts.header.name}\"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void (*infer_fn)(uint8_t *, const int8_t *, int8_t *);

static uint64_t elapsed_ns(struct timespec start, struct timespec end) {{
    return (uint64_t)(end.tv_sec - start.tv_sec) * UINT64_C(1000000000)
        + (uint64_t)(end.tv_nsec - start.tv_nsec);
}}

int main(int argc, char **argv) {{
    if (argc != 3) {{ return 2; }}
    infer_fn infer = strcmp(argv[1], "optimized") == 0
        ? {optimized_symbol}_infer : {portable_symbol}_infer;
    const size_t iterations = (size_t)strtoull(argv[2], NULL, 10);
    int8_t input[{macro}_INPUT_SIZE];
    int8_t output[{macro}_OUTPUT_SIZE];
    memset(input, 1, sizeof(input));
    int64_t checksum = 0;
    for (size_t index = 0; index < 1000u; ++index) {{ infer(NULL, input, output); }}
    struct timespec start;
    struct timespec end;
    if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) {{ return 3; }}
    for (size_t index = 0; index < iterations; ++index) {{
        input[index % {macro}_INPUT_SIZE] = (int8_t)((int32_t)(index % 127u) - 63);
        infer(NULL, input, output);
        checksum += output[index % {macro}_OUTPUT_SIZE];
    }}
    if (clock_gettime(CLOCK_MONOTONIC, &end) != 0) {{ return 4; }}
    printf("%llu %lld\\n", (unsigned long long)elapsed_ns(start, end), (long long)checksum);
    return 0;
}}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--kernel", choices=tuple(_GRAPH_FACTORIES), default="linear")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--trials", type=int, default=7)
    args = parser.parse_args()
    if args.iterations is None:
        args.iterations = _DEFAULT_ITERATIONS[args.kernel]
    if args.iterations <= 0 or args.trials <= 0:
        parser.error("iterations and trials must be positive")

    with tempfile.TemporaryDirectory(prefix="bakenn-host-linear-") as temporary:
        root = Path(temporary)
        graph = _GRAPH_FACTORIES[args.kernel]()
        portable = bakenn.compile(graph, root / "portable", model_name="host_portable")
        optimized = bakenn.compile(
            graph,
            root / "optimized",
            model_name="host_optimized",
            backend_options=bakenn.CBackendOptions(
                kernel_policy=bakenn.KernelPolicy.REQUIRE_OPTIMIZED
            ),
        )
        runner = root / "runner.c"
        runner.write_text(_runner(portable, optimized), encoding="utf-8")
        executable = root / "runner"
        sources = [
            portable.artifacts.model_source,
            portable.artifacts.weights_source,
            portable.artifacts.kernels_source,
            optimized.artifacts.model_source,
            optimized.artifacts.weights_source,
            optimized.artifacts.kernels_source,
            runner,
        ]
        subprocess.run(
            [
                args.cc,
                "-std=c11",
                "-O3",
                "-DNDEBUG",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-pedantic",
                *(str(path) for path in sources),
                "-I",
                str(portable.artifacts.output_dir),
                "-I",
                str(optimized.artifacts.output_dir),
                "-o",
                str(executable),
            ],
            check=True,
        )
        timings: dict[str, list[float]] = {"portable": [], "optimized": []}
        checksums: dict[str, set[int]] = {"portable": set(), "optimized": set()}
        for trial in range(args.trials):
            order = ("portable", "optimized") if trial % 2 == 0 else ("optimized", "portable")
            for implementation in order:
                output = subprocess.check_output(
                    [str(executable), implementation, str(args.iterations)], text=True
                ).strip()
                elapsed_text, checksum_text = output.split()
                timings[implementation].append(int(elapsed_text) / args.iterations)
                checksums[implementation].add(int(checksum_text))
        if checksums["portable"] != checksums["optimized"]:
            raise RuntimeError("portable and optimized benchmark checksums differ")
        portable_ns = statistics.median(timings["portable"])
        optimized_ns = statistics.median(timings["optimized"])
        result = {
            "status": "host_smoke_not_mcu_evidence",
            "compiler": args.cc,
            "compiler_version": subprocess.check_output(
                [args.cc, "--version"], text=True
            ).splitlines()[0],
            "kernel": args.kernel,
            "input_shape": list(graph.values[graph.inputs[0]].shape),
            "output_shape": list(graph.values[graph.outputs[0]].shape),
            "iterations_per_trial": args.iterations,
            "trials": args.trials,
            "portable_median_ns": portable_ns,
            "optimized_median_ns": optimized_ns,
            "portable_over_optimized": portable_ns / optimized_ns,
            "checksum": next(iter(checksums["portable"])),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
