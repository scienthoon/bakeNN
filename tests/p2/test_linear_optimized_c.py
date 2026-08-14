from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

import bakenn
from bakenn.ir import (
    DType,
    Layout,
    LinearOp,
    PerAxisQParams,
    PerTensorQParams,
    QuantizedGraph,
    TensorType,
)

from .test_backend_selection import linear_graph
from .support import require_compiler


def _runner_source(portable: bakenn.compiler.CompiledModel, optimized: bakenn.compiler.CompiledModel) -> str:
    portable_manifest = json.loads(portable.artifacts.manifest.read_text(encoding="utf-8"))
    optimized_manifest = json.loads(optimized.artifacts.manifest.read_text(encoding="utf-8"))
    portable_symbol = portable_manifest["model"]
    optimized_symbol = optimized_manifest["model"]
    portable_macro = portable_symbol.upper()
    optimized_macro = optimized_symbol.upper()
    return f"""#include \"{portable.artifacts.header.name}\"
#include \"{optimized.artifacts.header.name}\"
#include <stdio.h>

int main(void) {{
    int8_t input[{portable_macro}_INPUT_SIZE];
    int8_t portable_output[{portable_macro}_OUTPUT_SIZE];
    int8_t optimized_output[{optimized_macro}_OUTPUT_SIZE];
    while (fread(input, 1u, {portable_macro}_INPUT_SIZE, stdin) == {portable_macro}_INPUT_SIZE) {{
        {portable_symbol}_infer(NULL, input, portable_output);
        {optimized_symbol}_infer(NULL, input, optimized_output);
        if (fwrite(portable_output, 1u, {portable_macro}_OUTPUT_SIZE, stdout)
                != {portable_macro}_OUTPUT_SIZE
            || fwrite(optimized_output, 1u, {optimized_macro}_OUTPUT_SIZE, stdout)
                != {optimized_macro}_OUTPUT_SIZE) {{
            return 2;
        }}
    }}
    return ferror(stdin) ? 3 : 0;
}}
"""


def _compile_pair(
    graph: QuantizedGraph,
    tmp_path: Path,
    *,
    compiler: str,
    optimization: str,
    prefix: str,
):  # type: ignore[no-untyped-def]
    portable = bakenn.compile(
        graph, tmp_path / f"{prefix}_portable", model_name=f"{prefix}_portable"
    )
    optimized = bakenn.compile(
        graph,
        tmp_path / f"{prefix}_optimized",
        model_name=f"{prefix}_optimized",
        backend_options=bakenn.CBackendOptions(
            kernel_policy=bakenn.KernelPolicy.AUTO
        ),
    )
    runner = tmp_path / f"{prefix}_runner.c"
    runner.write_text(_runner_source(portable, optimized), encoding="utf-8")
    executable = tmp_path / f"{prefix}_{optimization[2:]}"
    subprocess.run(
        [
            compiler,
            "-std=c11",
            optimization,
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all",
            *(str(path) for path in (
                portable.artifacts.model_source,
                portable.artifacts.weights_source,
                portable.artifacts.kernels_source,
                optimized.artifacts.model_source,
                optimized.artifacts.weights_source,
                optimized.artifacts.kernels_source,
                runner,
            )),
            "-I",
            str(portable.artifacts.output_dir),
            "-I",
            str(optimized.artifacts.output_dir),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
    )
    return portable, optimized, executable


def _assert_differential(
    portable: bakenn.compiler.CompiledModel,
    executable: Path,
    inputs: np.ndarray,
) -> None:
    input_shape = portable.plan.tensors[portable.plan.inputs[0]].tensor_type.shape
    output_shape = portable.plan.tensors[portable.plan.outputs[0]].tensor_type.shape
    expected = np.concatenate(
        [bakenn.run_reference(portable.plan, row.reshape(input_shape)) for row in inputs],
        axis=0,
    )
    result = subprocess.run(
        executable, input=inputs.tobytes(), capture_output=True, check=True
    )
    actual = np.frombuffer(result.stdout, dtype=np.int8).reshape(
        len(inputs), 2, *output_shape[1:]
    )
    np.testing.assert_array_equal(actual[:, 0], expected)
    np.testing.assert_array_equal(actual[:, 1], expected)


@pytest.mark.parametrize("optimization", ["-O0", "-O2", "-Os"])
def test_optimized_linear_is_python_and_portable_c_bit_exact(
    tmp_path: Path, optimization: str
) -> None:
    compiler = os.environ.get("CC", "cc")
    compiler = require_compiler(compiler)
    graph = linear_graph()
    portable = bakenn.compile(graph, tmp_path / "portable", model_name="linear_portable")
    optimized = bakenn.compile(
        graph,
        tmp_path / "optimized",
        model_name="linear_optimized",
        backend_options=bakenn.CBackendOptions(kernel_policy=bakenn.KernelPolicy.AUTO),
    )
    assert portable.plan is not optimized.plan
    assert portable.plan.steps == optimized.plan.steps
    assert optimized.artifacts.backend_plan.selections[0].optimized

    runner = tmp_path / "runner.c"
    runner.write_text(_runner_source(portable, optimized), encoding="utf-8")
    executable = tmp_path / f"runner_{optimization[2:]}"
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
            compiler,
            "-std=c11",
            optimization,
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all",
            *(str(path) for path in sources),
            "-I",
            str(portable.artifacts.output_dir),
            "-I",
            str(optimized.artifacts.output_dir),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
    )

    rng = np.random.default_rng(20260814)
    edges = np.asarray(
        [
            [-128] * 12,
            [127] * 12,
            [-7] * 12,
            [-128, 127, -7, 0, 1, -1, 64, -64, 126, -127, 2, -2],
        ],
        dtype=np.int8,
    )
    random_inputs = rng.integers(-128, 128, size=(512, 12), dtype=np.int16).astype(np.int8)
    inputs = np.concatenate((edges, random_inputs), axis=0)
    expected = np.concatenate(
        [bakenn.run_reference(portable.plan, row.reshape(1, -1)) for row in inputs], axis=0
    )
    result = subprocess.run(executable, input=inputs.tobytes(), capture_output=True, check=True)
    actual = np.frombuffer(result.stdout, dtype=np.int8).reshape(len(inputs), 2, 6)
    np.testing.assert_array_equal(actual[:, 0, :], expected)
    np.testing.assert_array_equal(actual[:, 1, :], expected)

    optimized_source = optimized.artifacts.kernels_source.read_text(encoding="utf-8")
    assert optimized_source.count("int32_t bknn_linear_optimized_q31_high_mul(") == 1
    assert "elementwise_high_mul" not in optimized_source
    assert "linear_high_mul" not in optimized_source


@pytest.mark.parametrize("optimization", ["-O0", "-O2", "-Os"])
def test_optimized_linear_tail_is_python_and_portable_c_bit_exact(
    tmp_path: Path, optimization: str
) -> None:
    compiler = os.environ.get("CC", "cc")
    compiler = require_compiler(compiler)
    graph = linear_graph(12, 5)
    portable = bakenn.compile(graph, tmp_path / "portable", model_name="linear_tail_portable")
    optimized = bakenn.compile(
        graph,
        tmp_path / "optimized",
        model_name="linear_tail_optimized",
        backend_options=bakenn.CBackendOptions(kernel_policy=bakenn.KernelPolicy.AUTO),
    )
    assert optimized.artifacts.backend_plan.selections[0].kernel_id == (
        "optimized.linear_oi2_tail.v1"
    )
    runner = tmp_path / "runner.c"
    runner.write_text(_runner_source(portable, optimized), encoding="utf-8")
    executable = tmp_path / f"runner_tail_{optimization[2:]}"
    subprocess.run(
        [
            compiler,
            "-std=c11",
            optimization,
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all",
            *(str(path) for path in (
                portable.artifacts.model_source,
                portable.artifacts.weights_source,
                portable.artifacts.kernels_source,
                optimized.artifacts.model_source,
                optimized.artifacts.weights_source,
                optimized.artifacts.kernels_source,
                runner,
            )),
            "-I",
            str(portable.artifacts.output_dir),
            "-I",
            str(optimized.artifacts.output_dir),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
    )
    rng = np.random.default_rng(20260815)
    inputs = rng.integers(-128, 128, size=(512, 12), dtype=np.int16).astype(np.int8)
    inputs[:4] = np.asarray(
        [[-128] * 12, [127] * 12, [-7] * 12, list(range(-6, 6))], dtype=np.int8
    )
    expected = np.concatenate(
        [bakenn.run_reference(portable.plan, row.reshape(1, -1)) for row in inputs], axis=0
    )
    result = subprocess.run(executable, input=inputs.tobytes(), capture_output=True, check=True)
    actual = np.frombuffer(result.stdout, dtype=np.int8).reshape(len(inputs), 2, 5)
    np.testing.assert_array_equal(actual[:, 0], expected)
    np.testing.assert_array_equal(actual[:, 1], expected)


def test_linear_optimized_kernel_has_10k_byte_exact_differential(tmp_path: Path) -> None:
    compiler = require_compiler(os.environ.get("CC", "cc"))
    graph = linear_graph(31, 8)
    portable, optimized, executable = _compile_pair(
        graph,
        tmp_path,
        compiler=compiler,
        optimization="-O2",
        prefix="linear_10k",
    )
    assert optimized.artifacts.backend_plan.selections[0].kernel_id == (
        "optimized.linear_oi2.v1"
    )
    assert all(shift > 0 for shift in portable.plan.steps[0].shifts)
    rng = np.random.default_rng(20260818)
    inputs = rng.integers(-128, 128, size=(10_000, 31), dtype=np.int16).astype(
        np.int8
    )
    inputs[:4] = np.asarray(
        [
            [-128] * 31,
            [127] * 31,
            [-7] * 31,
            ((np.arange(31, dtype=np.int16) * 37) % 256 - 128).astype(np.int8),
        ],
        dtype=np.int8,
    )
    _assert_differential(portable, executable, inputs)


def _int32_boundary_graph() -> QuantizedGraph:
    input_count = 66_311
    input_q = PerTensorQParams(0.125, 127)
    weight_q = PerAxisQParams((0.125, 0.125), (0, 0), 0)
    output_q = PerTensorQParams(0.25, 0)
    weight = np.full((2, input_count), 127, dtype=np.int8)
    bias = np.zeros(2, dtype=np.int32)
    return QuantizedGraph(
        name="linear_int32_boundary",
        values={
            "input": TensorType(
                (1, input_count), DType.INT8, Layout.NC, input_q
            ),
            "weight": TensorType(
                (2, input_count), DType.INT8, Layout.OI, weight_q
            ),
            "bias": TensorType(
                (2,),
                DType.INT32,
                Layout.C,
                PerAxisQParams((0.015625, 0.015625), (0, 0), 0),
            ),
            "output": TensorType((1, 2), DType.INT8, Layout.NC, output_q),
        },
        constants={"weight": weight, "bias": bias},
        ops=(LinearOp("linear", "input", "weight", "bias", "output"),),
        inputs=("input",),
        outputs=("output",),
    )


def test_linear_optimized_accumulator_near_int32_boundary(tmp_path: Path) -> None:
    compiler = require_compiler(os.environ.get("CC", "cc"))
    graph = _int32_boundary_graph()
    portable, optimized, executable = _compile_pair(
        graph,
        tmp_path,
        compiler=compiler,
        optimization="-O2",
        prefix="linear_int32_boundary",
    )
    assert optimized.artifacts.backend_plan.selections[0].optimized
    bound = portable.plan.steps[0].accumulator_bounds[0]
    assert bound == 2_147_481_735
    assert bound < (1 << 31) - 1
    inputs = np.stack(
        (
            np.full(66_311, -128, dtype=np.int8),
            np.full(66_311, 127, dtype=np.int8),
        )
    )
    _assert_differential(portable, executable, inputs)
