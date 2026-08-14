from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from bakenn.backend.portable_c.generator import generate_portable_c
from bakenn.ir.graph import QuantizedGraph
from bakenn.ir.ops.conv import Conv2DOp, DepthwiseConv2DOp
from bakenn.ir.types import DType, Layout, PerAxisQParams, PerTensorQParams, TensorType
from bakenn.plan.lower import lower_to_plan
from bakenn.reference.executor import run_reference

# Explicit work-package registration imports; integration owns aggregators.
import bakenn.backend.portable_c.families.conv  # noqa: F401
import bakenn.ir.verifiers.conv  # noqa: F401
import bakenn.plan.lowering.conv  # noqa: F401
import bakenn.reference.kernels.conv  # noqa: F401


def _random_graph(rng: np.random.Generator, *, depthwise: bool) -> QuantizedGraph:
    input_height = int(rng.integers(2, 6))
    input_width = int(rng.integers(2, 6))
    input_channels = int(rng.integers(1, 4))
    kernel_height = int(rng.integers(1, 4))
    kernel_width = int(rng.integers(1, 4))
    stride = (int(rng.integers(1, 3)), int(rng.integers(1, 3)))
    dilation = (int(rng.integers(1, 3)), int(rng.integers(1, 3)))
    padding = tuple(int(value) for value in rng.integers(0, 3, size=4))
    effective_height = dilation[0] * (kernel_height - 1) + 1
    effective_width = dilation[1] * (kernel_width - 1) + 1
    if input_height + padding[0] + padding[1] < effective_height:
        padding = (
            padding[0],
            effective_height - input_height - padding[0],
            padding[2],
            padding[3],
        )
    if input_width + padding[2] + padding[3] < effective_width:
        padding = (
            padding[0],
            padding[1],
            padding[2],
            effective_width - input_width - padding[2],
        )
    output_height = (
        input_height + padding[0] + padding[1] - effective_height
    ) // stride[0] + 1
    output_width = (
        input_width + padding[2] + padding[3] - effective_width
    ) // stride[1] + 1
    input_scale = float(rng.uniform(0.01, 0.3))
    output_scale = float(rng.uniform(0.04, 0.8))
    input_zero_point = int(rng.integers(-100, 101))
    output_zero_point = int(rng.integers(-100, 101))

    if depthwise:
        depth_multiplier = int(rng.integers(1, 4))
        output_channels = input_channels * depth_multiplier
        weight_shape = (kernel_height, kernel_width, output_channels)
        weight_layout = Layout.HWO
        weight_axis = 2
        op_type = DepthwiseConv2DOp
    else:
        depth_multiplier = 1
        output_channels = int(rng.integers(1, 5))
        weight_shape = (output_channels, kernel_height, kernel_width, input_channels)
        weight_layout = Layout.OHWI
        weight_axis = 0
        op_type = Conv2DOp
    weight_scales = tuple(float(value) for value in rng.uniform(0.01, 0.2, output_channels))
    input_qparams = PerTensorQParams(input_scale, input_zero_point)
    output_qparams = PerTensorQParams(output_scale, output_zero_point)
    weight_qparams = PerAxisQParams(weight_scales, (0,) * output_channels, weight_axis)
    bias_qparams = PerAxisQParams(
        tuple(input_qparams.scale * scale for scale in weight_qparams.scales),
        (0,) * output_channels,
        0,
    )
    weight = rng.integers(-25, 26, size=weight_shape, dtype=np.int16).astype(np.int8)
    bias = rng.integers(-300, 301, size=output_channels, dtype=np.int32)
    activation_min = int(rng.integers(-128, 1))
    activation_max = int(rng.integers(max(activation_min, 0), 128))
    common = dict(
        name="conv",
        input="input",
        weight="weight",
        bias="bias",
        output="output",
        stride=stride,
        dilation=dilation,
        padding=padding,
        activation_min=activation_min,
        activation_max=activation_max,
    )
    op = op_type(**common, depth_multiplier=depth_multiplier) if depthwise else op_type(**common)
    return QuantizedGraph(
        name="random_depthwise" if depthwise else "random_conv",
        values={
            "input": TensorType(
                (1, input_height, input_width, input_channels),
                DType.INT8,
                Layout.NHWC,
                input_qparams,
            ),
            "weight": TensorType(weight_shape, DType.INT8, weight_layout, weight_qparams),
            "bias": TensorType((output_channels,), DType.INT32, Layout.C, bias_qparams),
            "output": TensorType(
                (1, output_height, output_width, output_channels),
                DType.INT8,
                Layout.NHWC,
                output_qparams,
            ),
        },
        constants={"weight": weight, "bias": bias},
        ops=(op,),
        inputs=("input",),
        outputs=("output",),
    )


def _compile_runner(plan, directory: Path, compiler: str) -> tuple[Path, object]:  # type: ignore[no-untyped-def]
    artifacts = generate_portable_c(plan, directory)
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    symbol = manifest["model"]
    macro = symbol.upper()
    runner = directory / "runner.c"
    runner.write_text(
        f"""#include "{artifacts.header.name}"
#include <stddef.h>
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
    _Alignas({macro}_ARENA_ALIGNMENT)
        uint8_t arena_storage[GUARD_SIZE + {macro}_ARENA_SIZE + GUARD_SIZE];
    struct {{ uint8_t before[GUARD_SIZE]; int8_t data[{macro}_INPUT_SIZE]; uint8_t after[GUARD_SIZE]; }} input;
    struct {{ uint8_t before[GUARD_SIZE]; int8_t data[{macro}_OUTPUT_SIZE]; uint8_t after[GUARD_SIZE]; }} output;
    memset(arena_storage, 0xA5, sizeof(arena_storage));
    memset(&input, 0xA5, sizeof(input));
    memset(&output, 0xA5, sizeof(output));
    uint8_t *arena = {macro}_ARENA_SIZE == 0u ? NULL : arena_storage + GUARD_SIZE;
    while (fread(input.data, 1u, {macro}_INPUT_SIZE, stdin) == {macro}_INPUT_SIZE) {{
        {symbol}_infer(arena, input.data, output.data);
        if (!guard_ok(arena_storage)
            || !guard_ok(arena_storage + GUARD_SIZE + {macro}_ARENA_SIZE)
            || !guard_ok(input.before) || !guard_ok(input.after)
            || !guard_ok(output.before) || !guard_ok(output.after)) {{ return 4; }}
        if (fwrite(output.data, 1u, {macro}_OUTPUT_SIZE, stdout) != {macro}_OUTPUT_SIZE) {{ return 2; }}
    }}
    return ferror(stdin) ? 3 : 0;
}}
""",
        encoding="utf-8",
    )
    executable = directory / "runner"
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-O1",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all",
            str(artifacts.model_source),
            str(artifacts.weights_source),
            str(artifacts.kernels_source),
            str(runner),
            "-I",
            str(directory),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
    )
    return executable, artifacts


@pytest.mark.parametrize("depthwise", [False, True])
@pytest.mark.parametrize("compiler", ["gcc", "clang"])
def test_randomized_conv_python_c_bit_exact_with_sanitizers(
    tmp_path: Path,
    depthwise: bool,
    compiler: str,
) -> None:
    if shutil.which(compiler) is None:
        pytest.skip(f"{compiler} is not installed")
    rng = np.random.default_rng(20260814 + int(depthwise))
    graph = _random_graph(rng, depthwise=depthwise)
    plan = lower_to_plan(graph)
    executable, artifacts = _compile_runner(plan, tmp_path / f"{compiler}_{depthwise}", compiler)

    # At least 256 randomized tensor inputs per operator/backend/compiler.
    input_shape = graph.values["input"].shape
    inputs = rng.integers(-128, 128, size=(256, *input_shape[1:]), dtype=np.int16).astype(np.int8)
    expected = np.concatenate(
        [run_reference(plan, sample.reshape(input_shape)) for sample in inputs], axis=0
    )
    result = subprocess.run(executable, input=inputs.tobytes(), capture_output=True, check=True)
    actual = np.frombuffer(result.stdout, dtype=np.int8).reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected)

    generated = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (artifacts.model_source, artifacts.weights_source, artifacts.kernels_source)
    )
    for forbidden in ("malloc(", "calloc(", "realloc(", "free(", "float ", "double "):
        assert forbidden not in generated


def test_conv_c_emits_asymmetric_padding_and_per_channel_parameters(tmp_path: Path) -> None:
    compiler = os.environ.get("CC", "cc")
    if shutil.which(compiler) is None:
        pytest.skip(f"{compiler} is not installed")
    rng = np.random.default_rng(7)
    graph = _random_graph(rng, depthwise=False)
    plan = lower_to_plan(graph)
    _, artifacts = _compile_runner(plan, tmp_path / "inspect", compiler)
    source = artifacts.model_source.read_text(encoding="utf-8")
    weights = artifacts.weights_source.read_text(encoding="utf-8")
    kernel = artifacts.kernels_source.read_text(encoding="utf-8")
    input_zp = graph.values["input"].qparams.zero_point
    assert f"        {input_zp}," in source
    assert "_op0_multiplier" in weights
    assert "_op0_shift" in weights
    assert "input_value = input_zero_point" in kernel
    assert "kernel_y * (int64_t)dilation_height" in kernel
