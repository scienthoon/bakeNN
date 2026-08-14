from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

import bakenn
from bakenn.errors import CompileError
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
from bakenn.plan import lower_to_plan
from bakenn.backend.portable_c import select_backend_plan
from .support import require_compiler


def _output_extent(size: int, kernel: int, stride: int, padding: tuple[int, int]) -> int:
    return (size + padding[0] + padding[1] - kernel) // stride + 1


def conv1x1_graph(*, output_channels: int = 4, stride: tuple[int, int] = (1, 1)) -> QuantizedGraph:
    input_shape = (1, 5, 6, 3)
    output_shape = (
        1,
        _output_extent(input_shape[1], 1, stride[0], (0, 0)),
        _output_extent(input_shape[2], 1, stride[1], (0, 0)),
        output_channels,
    )
    input_q = PerTensorQParams(0.125, 127)
    output_q = PerTensorQParams(0.25, 3)
    scales = tuple(0.05 + 0.01 * channel for channel in range(output_channels))
    weight = (
        (np.arange(output_channels * 3, dtype=np.int16) * 11 + 5) % 31 - 15
    ).reshape(output_channels, 1, 1, 3).astype(np.int8)
    bias = (np.arange(output_channels, dtype=np.int32) * 23 - 31).astype(np.int32)
    return QuantizedGraph(
        name=f"p2_conv1x1_{output_channels}_{stride[0]}{stride[1]}",
        values={
            "input": TensorType(input_shape, DType.INT8, Layout.NHWC, input_q),
            "weight": TensorType(
                weight.shape,
                DType.INT8,
                Layout.OHWI,
                PerAxisQParams(scales, (0,) * output_channels, 0),
            ),
            "bias": TensorType(
                (output_channels,),
                DType.INT32,
                Layout.C,
                PerAxisQParams(
                    tuple(input_q.scale * scale for scale in scales),
                    (0,) * output_channels,
                    0,
                ),
            ),
            "output": TensorType(output_shape, DType.INT8, Layout.NHWC, output_q),
        },
        constants={"weight": weight, "bias": bias},
        ops=(
            Conv2DOp(
                "conv1x1",
                "input",
                "weight",
                "bias",
                "output",
                stride=stride,
                activation_min=-13,
                activation_max=107,
            ),
        ),
        inputs=("input",),
        outputs=("output",),
    )


def depthwise3x3_graph(
    *, input_channels: int = 4, stride: tuple[int, int] = (1, 1), padding=(1, 1, 1, 1)
) -> QuantizedGraph:
    input_shape = (1, 5, 6, input_channels)
    output_shape = (
        1,
        _output_extent(input_shape[1], 3, stride[0], (padding[0], padding[1])),
        _output_extent(input_shape[2], 3, stride[1], (padding[2], padding[3])),
        input_channels,
    )
    input_q = PerTensorQParams(0.0625, 127)
    output_q = PerTensorQParams(0.125, -11)
    scales = tuple(0.04 + 0.005 * channel for channel in range(input_channels))
    weight = (
        (np.arange(9 * input_channels, dtype=np.int16) * 13 + 7) % 29 - 14
    ).reshape(3, 3, input_channels).astype(np.int8)
    bias = (np.arange(input_channels, dtype=np.int32) * 17 - 23).astype(np.int32)
    return QuantizedGraph(
        name=f"p2_depthwise3x3_{input_channels}_{stride[0]}{stride[1]}",
        values={
            "input": TensorType(input_shape, DType.INT8, Layout.NHWC, input_q),
            "weight": TensorType(
                weight.shape,
                DType.INT8,
                Layout.HWO,
                PerAxisQParams(scales, (0,) * input_channels, 2),
            ),
            "bias": TensorType(
                (input_channels,),
                DType.INT32,
                Layout.C,
                PerAxisQParams(
                    tuple(input_q.scale * scale for scale in scales),
                    (0,) * input_channels,
                    0,
                ),
            ),
            "output": TensorType(output_shape, DType.INT8, Layout.NHWC, output_q),
        },
        constants={"weight": weight, "bias": bias},
        ops=(
            DepthwiseConv2DOp(
                "depthwise3x3",
                "input",
                "weight",
                "bias",
                "output",
                depth_multiplier=1,
                stride=stride,
                padding=padding,
                activation_min=-87,
                activation_max=99,
            ),
        ),
        inputs=("input",),
        outputs=("output",),
    )


def _with_activation_qparams(
    graph: QuantizedGraph,
    *,
    input_q: PerTensorQParams,
    output_q: PerTensorQParams,
) -> QuantizedGraph:
    values = dict(graph.values)
    input_type = values["input"]
    output_type = values["output"]
    weight_qparams = values["weight"].qparams
    assert isinstance(weight_qparams, PerAxisQParams)
    values["input"] = TensorType(
        input_type.shape, input_type.dtype, input_type.layout, input_q
    )
    values["output"] = TensorType(
        output_type.shape, output_type.dtype, output_type.layout, output_q
    )
    bias_type = values["bias"]
    values["bias"] = TensorType(
        bias_type.shape,
        bias_type.dtype,
        bias_type.layout,
        PerAxisQParams(
            tuple(input_q.scale * scale for scale in weight_qparams.scales),
            weight_qparams.zero_points,
            0,
        ),
    )
    return QuantizedGraph(
        name=f"{graph.name}_requantized",
        values=values,
        constants=graph.constants,
        ops=graph.ops,
        inputs=graph.inputs,
        outputs=graph.outputs,
    )


def _runner_source(portable, optimized) -> str:  # type: ignore[no-untyped-def]
    portable_manifest = json.loads(portable.artifacts.manifest.read_text(encoding="utf-8"))
    optimized_manifest = json.loads(optimized.artifacts.manifest.read_text(encoding="utf-8"))
    portable_symbol = portable_manifest["model"]
    optimized_symbol = optimized_manifest["model"]
    portable_macro = portable_symbol.upper()
    optimized_macro = optimized_symbol.upper()
    return f"""#include \"{portable.artifacts.header.name}\"
#include \"{optimized.artifacts.header.name}\"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define GUARD_SIZE 32u

static int guard_ok(const uint8_t *guard) {{
    for (size_t index = 0; index < GUARD_SIZE; ++index) {{
        if (guard[index] != UINT8_C(0xA5)) {{ return 0; }}
    }}
    return 1;
}}

int main(void) {{
    _Alignas({portable_macro}_ARENA_ALIGNMENT)
        uint8_t arena_a[GUARD_SIZE + {portable_macro}_ARENA_SIZE + GUARD_SIZE];
    _Alignas({optimized_macro}_ARENA_ALIGNMENT)
        uint8_t arena_b[GUARD_SIZE + {optimized_macro}_ARENA_SIZE + GUARD_SIZE];
    struct {{ uint8_t before[GUARD_SIZE]; int8_t data[{portable_macro}_INPUT_SIZE]; uint8_t after[GUARD_SIZE]; }} input;
    struct {{ uint8_t before[GUARD_SIZE]; int8_t data[{portable_macro}_OUTPUT_SIZE]; uint8_t after[GUARD_SIZE]; }} output_a;
    struct {{ uint8_t before[GUARD_SIZE]; int8_t data[{optimized_macro}_OUTPUT_SIZE]; uint8_t after[GUARD_SIZE]; }} output_b;
    memset(arena_a, 0xA5, sizeof(arena_a));
    memset(arena_b, 0xA5, sizeof(arena_b));
    memset(&input, 0xA5, sizeof(input));
    memset(&output_a, 0xA5, sizeof(output_a));
    memset(&output_b, 0xA5, sizeof(output_b));
    uint8_t *arena_ptr_a = {portable_macro}_ARENA_SIZE == 0u ? NULL : arena_a + GUARD_SIZE;
    uint8_t *arena_ptr_b = {optimized_macro}_ARENA_SIZE == 0u ? NULL : arena_b + GUARD_SIZE;
    while (fread(input.data, 1u, {portable_macro}_INPUT_SIZE, stdin) == {portable_macro}_INPUT_SIZE) {{
        {portable_symbol}_infer(arena_ptr_a, input.data, output_a.data);
        {optimized_symbol}_infer(arena_ptr_b, input.data, output_b.data);
        if (!guard_ok(arena_a) || !guard_ok(arena_a + GUARD_SIZE + {portable_macro}_ARENA_SIZE)
            || !guard_ok(arena_b) || !guard_ok(arena_b + GUARD_SIZE + {optimized_macro}_ARENA_SIZE)
            || !guard_ok(input.before) || !guard_ok(input.after)
            || !guard_ok(output_a.before) || !guard_ok(output_a.after)
            || !guard_ok(output_b.before) || !guard_ok(output_b.after)) {{ return 4; }}
        if (fwrite(output_a.data, 1u, {portable_macro}_OUTPUT_SIZE, stdout) != {portable_macro}_OUTPUT_SIZE
            || fwrite(output_b.data, 1u, {optimized_macro}_OUTPUT_SIZE, stdout) != {optimized_macro}_OUTPUT_SIZE) {{ return 2; }}
    }}
    return ferror(stdin) ? 3 : 0;
}}
"""


def _compile_pair(
    graph: QuantizedGraph, tmp_path: Path, compiler: str, optimization: str
):
    portable = bakenn.compile(graph, tmp_path / "portable", model_name="portable")
    optimized = bakenn.compile(
        graph,
        tmp_path / "optimized",
        model_name="optimized",
        backend_options=bakenn.CBackendOptions(kernel_policy=bakenn.KernelPolicy.AUTO),
    )
    runner = tmp_path / f"runner_{optimization[2:]}.c"
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
    return portable, optimized, executable


def _assert_differential(
    graph: QuantizedGraph,
    portable,  # type: ignore[no-untyped-def]
    executable: Path,
    inputs: np.ndarray,
) -> None:
    input_shape = graph.values["input"].shape
    output_shape = graph.values["output"].shape
    expected = np.concatenate(
        [
            bakenn.run_reference(portable.plan, value.reshape(input_shape))
            for value in inputs
        ],
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


@pytest.mark.parametrize("stride", [(1, 1), (2, 2)])
@pytest.mark.parametrize("compiler", ["gcc", "clang"])
@pytest.mark.parametrize("optimization", ["-O0", "-O2", "-Os"])
def test_conv1x1_optimized_matches_portable_and_reference(
    tmp_path: Path, stride: tuple[int, int], compiler: str, optimization: str
) -> None:
    compiler = require_compiler(compiler)
    graph = conv1x1_graph(stride=stride)
    plan = lower_to_plan(graph)
    backend = select_backend_plan(
        plan, bakenn.CBackendOptions(kernel_policy=bakenn.KernelPolicy.AUTO)
    )
    assert backend.selections[0].kernel_id == "optimized.conv2d_1x1_o2.v1"
    assert backend.packed_constants["weight.conv2d_1x1_o2"].layout == (
        "conv2d_1x1_ohwi_o2_interleaved_v1"
    )
    assert all(shift < 0 for shift in plan.steps[0].shifts)
    portable, optimized, executable = _compile_pair(
        graph, tmp_path, compiler, optimization
    )
    assert optimized.artifacts.backend_plan.selections[0].optimized
    rng = np.random.default_rng(20260816)
    inputs = rng.integers(-128, 128, size=(256, *graph.values["input"].shape[1:]), dtype=np.int16).astype(np.int8)
    inputs[:4] = np.asarray(
        [
            np.full(inputs.shape[1:], -128),
            np.full(inputs.shape[1:], 127),
            np.full(inputs.shape[1:], graph.values["input"].qparams.zero_point),
            np.arange(np.prod(inputs.shape[1:]), dtype=np.int16).reshape(inputs.shape[1:]) - 20,
        ],
        dtype=np.int8,
    )
    expected = np.concatenate(
        [bakenn.run_reference(portable.plan, value.reshape(graph.values["input"].shape)) for value in inputs],
        axis=0,
    )
    result = subprocess.run(executable, input=inputs.tobytes(), capture_output=True, check=True)
    actual = np.frombuffer(result.stdout, dtype=np.int8).reshape(len(inputs), 2, *graph.values["output"].shape[1:])
    np.testing.assert_array_equal(actual[:, 0], expected)
    np.testing.assert_array_equal(actual[:, 1], expected)


@pytest.mark.parametrize("compiler", ["gcc", "clang"])
@pytest.mark.parametrize("optimization", ["-O0", "-O2", "-Os"])
def test_depthwise_3x3_optimized_matches_portable_and_reference(
    tmp_path: Path, compiler: str, optimization: str
) -> None:
    compiler = require_compiler(compiler)
    graph = depthwise3x3_graph(stride=(2, 2))
    plan = lower_to_plan(graph)
    backend = select_backend_plan(
        plan, bakenn.CBackendOptions(kernel_policy=bakenn.KernelPolicy.AUTO)
    )
    assert backend.selections[0].kernel_id == "optimized.depthwise_3x3_c2.v1"
    assert backend.packed_constants["weight.depthwise_3x3_c2"].layout == (
        "depthwise_hwo_c2_interleaved_v1"
    )
    assert all(shift < 0 for shift in plan.steps[0].shifts)
    portable, optimized, executable = _compile_pair(
        graph, tmp_path, compiler, optimization
    )
    assert optimized.artifacts.backend_plan.selections[0].optimized
    rng = np.random.default_rng(20260817)
    inputs = rng.integers(-128, 128, size=(256, *graph.values["input"].shape[1:]), dtype=np.int16).astype(np.int8)
    inputs[:3] = np.asarray(
        [
            np.full(inputs.shape[1:], -128),
            np.full(inputs.shape[1:], 127),
            np.full(inputs.shape[1:], 127),
        ],
        dtype=np.int8,
    )
    expected = np.concatenate(
        [bakenn.run_reference(portable.plan, value.reshape(graph.values["input"].shape)) for value in inputs],
        axis=0,
    )
    result = subprocess.run(executable, input=inputs.tobytes(), capture_output=True, check=True)
    actual = np.frombuffer(result.stdout, dtype=np.int8).reshape(len(inputs), 2, *graph.values["output"].shape[1:])
    np.testing.assert_array_equal(actual[:, 0], expected)
    np.testing.assert_array_equal(actual[:, 1], expected)


def test_conv1x1_optimized_has_10k_positive_shift_differential(
    tmp_path: Path,
) -> None:
    compiler = require_compiler(os.environ.get("CC", "cc"))
    graph = _with_activation_qparams(
        conv1x1_graph(stride=(1, 2)),
        input_q=PerTensorQParams(0.5, -128),
        output_q=PerTensorQParams(0.01, 127),
    )
    portable, optimized, executable = _compile_pair(
        graph, tmp_path, compiler, "-O2"
    )
    assert optimized.artifacts.backend_plan.selections[0].kernel_id == (
        "optimized.conv2d_1x1_o2.v1"
    )
    assert all(shift > 0 for shift in portable.plan.steps[0].shifts)
    rng = np.random.default_rng(20260819)
    input_shape = graph.values["input"].shape
    inputs = rng.integers(
        -128,
        -119,
        size=(10_000, *input_shape[1:]),
        dtype=np.int16,
    ).astype(np.int8)
    inputs[:4] = np.asarray(
        [
            np.full(input_shape[1:], -128),
            np.full(input_shape[1:], 127),
            np.full(input_shape[1:], -127),
            (np.arange(np.prod(input_shape[1:])).reshape(input_shape[1:]) - 128),
        ],
        dtype=np.int8,
    )
    _assert_differential(graph, portable, executable, inputs)


def test_depthwise_optimized_has_10k_asymmetric_padding_differential(
    tmp_path: Path,
) -> None:
    compiler = require_compiler(os.environ.get("CC", "cc"))
    graph = _with_activation_qparams(
        depthwise3x3_graph(stride=(1, 2), padding=(0, 2, 1, 0)),
        input_q=PerTensorQParams(0.5, -128),
        output_q=PerTensorQParams(0.01, 127),
    )
    portable, optimized, executable = _compile_pair(
        graph, tmp_path, compiler, "-O2"
    )
    assert optimized.artifacts.backend_plan.selections[0].kernel_id == (
        "optimized.depthwise_3x3_c2.v1"
    )
    assert all(shift > 0 for shift in portable.plan.steps[0].shifts)
    rng = np.random.default_rng(20260820)
    input_shape = graph.values["input"].shape
    inputs = rng.integers(
        -128,
        -119,
        size=(10_000, *input_shape[1:]),
        dtype=np.int16,
    ).astype(np.int8)
    inputs[:4] = np.asarray(
        [
            np.full(input_shape[1:], -128),
            np.full(input_shape[1:], 127),
            np.full(input_shape[1:], -127),
            ((np.arange(np.prod(input_shape[1:])) * 37) % 256 - 128).reshape(
                input_shape[1:]
            ),
        ],
        dtype=np.int8,
    )
    _assert_differential(graph, portable, executable, inputs)


def test_conv_and_depthwise_optimized_candidates_fail_closed_on_unsupported_shapes() -> None:
    odd_conv = lower_to_plan(conv1x1_graph(output_channels=3))
    odd_backend = select_backend_plan(
        odd_conv, bakenn.CBackendOptions(kernel_policy=bakenn.KernelPolicy.AUTO)
    )
    assert odd_backend.selections[0].kernel_id == "portable.conv2d_s8.v1"
    assert "optimized.conv2d_1x1_o2.v1" in odd_backend.selections[0].rejected
    with pytest.raises(CompileError, match="no supported implementation"):
        select_backend_plan(
            odd_conv,
            bakenn.CBackendOptions(kernel_policy=bakenn.KernelPolicy.REQUIRE_OPTIMIZED),
        )

    general_conv_graph = conv1x1_graph()
    values = dict(general_conv_graph.values)
    weight = general_conv_graph.constants["weight"].repeat(9, axis=1).reshape(4, 3, 3, 3)
    values["weight"] = TensorType(
        weight.shape,
        DType.INT8,
        Layout.OHWI,
        PerAxisQParams((0.05, 0.06, 0.07, 0.08), (0, 0, 0, 0), 0),
    )
    values["output"] = TensorType((1, 5, 6, 4), DType.INT8, Layout.NHWC, values["output"].qparams)
    general = QuantizedGraph(
        name="p2_general_conv",
        values=values,
        constants={"weight": weight, "bias": general_conv_graph.constants["bias"]},
        ops=(
            Conv2DOp(
                "conv3x3", "input", "weight", "bias", "output", padding=(1, 1, 1, 1)
            ),
        ),
        inputs=("input",),
        outputs=("output",),
    )
    general_backend = select_backend_plan(
        lower_to_plan(general), bakenn.CBackendOptions(kernel_policy=bakenn.KernelPolicy.AUTO)
    )
    assert general_backend.selections[0].kernel_id == "portable.conv2d_s8.v1"
    with pytest.raises(CompileError, match="no supported implementation"):
        select_backend_plan(
            lower_to_plan(general),
            bakenn.CBackendOptions(kernel_policy=bakenn.KernelPolicy.REQUIRE_OPTIMIZED),
        )

    odd_depthwise = depthwise3x3_graph(input_channels=3)
    odd_depthwise_backend = select_backend_plan(
        lower_to_plan(odd_depthwise),
        bakenn.CBackendOptions(kernel_policy=bakenn.KernelPolicy.AUTO),
    )
    assert odd_depthwise_backend.selections[0].kernel_id == "portable.depthwise_conv2d_s8.v1"
    with pytest.raises(CompileError, match="no supported implementation"):
        select_backend_plan(
            lower_to_plan(odd_depthwise),
            bakenn.CBackendOptions(kernel_policy=bakenn.KernelPolicy.REQUIRE_OPTIMIZED),
        )
