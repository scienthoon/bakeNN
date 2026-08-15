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
    AveragePool2DOp,
    DType,
    Layout,
    MaxPool2DOp,
    PerTensorQParams,
    QuantizedGraph,
    TensorType,
)
from bakenn.targets import ESP32, ESP32_S3, export_esp_idf_project
from tests.p0.model_fixtures import residual_ds_cnn_graph
from tests.p2.test_backend_selection import linear_graph

from .support import require_compiler


def _options(target, *, policy=bakenn.KernelPolicy.AUTO):  # type: ignore[no-untyped-def]
    return bakenn.CBackendOptions(
        kernel_policy=policy,
        enable_esp_nn=True,
        target=target,
    )


def _pool_graph(
    op_type,  # type: ignore[no-untyped-def]
    *,
    zero_point: int = 0,
    kernel: tuple[int, int] = (3, 3),
) -> QuantizedGraph:
    qparams = PerTensorQParams(0.125, zero_point)
    output_height = 4 if kernel == (3, 3) else 3
    output_width = 5 if kernel == (3, 3) else 4
    return QuantizedGraph(
        name=f"esp_nn_{op_type.__name__}_{zero_point}_{kernel[0]}",
        values={
            "input": TensorType((1, 4, 5, 4), DType.INT8, Layout.NHWC, qparams),
            "output": TensorType(
                (1, output_height, output_width, 4),
                DType.INT8,
                Layout.NHWC,
                qparams,
            ),
        },
        constants={},
        ops=(
            op_type(
                "pool",
                "input",
                "output",
                kernel=kernel,
                stride=(1, 1),
                padding=(1, 1, 1, 1) if kernel == (3, 3) else (0, 0, 0, 0),
                activation_min=-91,
                activation_max=109,
            ),
        ),
        inputs=("input",),
        outputs=("output",),
    )


def _runner_source(compiled) -> str:  # type: ignore[no-untyped-def]
    manifest = json.loads(compiled.artifacts.manifest.read_text(encoding="utf-8"))
    symbol = str(manifest["model"])
    macro = symbol.upper()
    return f'''#include "{compiled.artifacts.header.name}"
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#define GUARD 32u

static int guard_ok(const uint8_t *value) {{
    for (size_t index = 0; index < GUARD; ++index) {{
        if (value[index] != UINT8_C(0xA5)) return 0;
    }}
    return 1;
}}

int main(void) {{
    _Alignas({macro}_ARENA_ALIGNMENT)
        uint8_t arena_storage[GUARD + {macro}_ARENA_SIZE + GUARD + 1u];
    struct {{ uint8_t before[GUARD]; int8_t data[{macro}_INPUT_SIZE]; uint8_t after[GUARD]; }} input;
    struct {{ uint8_t before[GUARD]; int8_t data[{macro}_OUTPUT_SIZE]; uint8_t after[GUARD]; }} output;
    memset(arena_storage, 0xA5, sizeof(arena_storage));
    memset(&input, 0xA5, sizeof(input));
    memset(&output, 0xA5, sizeof(output));
    uint8_t *arena = {macro}_ARENA_SIZE ? arena_storage + GUARD : NULL;
    while (fread(input.data, 1u, {macro}_INPUT_SIZE, stdin) == {macro}_INPUT_SIZE) {{
        {symbol}_infer(arena, input.data, output.data);
        if (!guard_ok(arena_storage)
            || !guard_ok(arena_storage + GUARD + {macro}_ARENA_SIZE)
            || !guard_ok(input.before) || !guard_ok(input.after)
            || !guard_ok(output.before) || !guard_ok(output.after)) return 4;
        if (fwrite(output.data, 1u, {macro}_OUTPUT_SIZE, stdout)
                != {macro}_OUTPUT_SIZE) return 2;
    }}
    return ferror(stdin) ? 3 : 0;
}}
'''


def _compile_host(
    graph: QuantizedGraph,
    output_dir: Path,
    target,
    compiler_name: str,
):  # type: ignore[no-untyped-def]
    compiler = require_compiler(compiler_name)
    compiled = bakenn.compile(
        graph,
        output_dir,
        model_name=f"esp_nn_{target.target_id}_{graph.name}",
        backend_options=_options(target),
        target=target,
    )
    runner = output_dir / "runner.c"
    runner.write_text(_runner_source(compiled), encoding="utf-8")
    executable = output_dir / "runner"
    support_sources = tuple(
        source
        for source in compiled.artifacts.support_sources
        if target is ESP32 or "esp32s3" not in source.name
    )
    definitions = ["-DCONFIG_NN_OPTIMIZED=1"] if target is ESP32 else []
    include_flags = [
        "-I",
        str(compiled.artifacts.output_dir),
        *(
            flag
            for include_dir in compiled.artifacts.support_include_dirs
            for flag in ("-I", str(include_dir))
        ),
    ]
    for source in (
        compiled.artifacts.model_source,
        compiled.artifacts.weights_source,
        compiled.artifacts.kernels_source,
        runner,
    ):
        subprocess.run(
            [
                compiler,
                "-std=c11",
                "-O2",
                *definitions,
                "-Wall",
                "-Wextra",
                "-Werror",
                "-pedantic",
                *include_flags,
                "-c",
                str(source),
                "-o",
                str(output_dir / f"strict_{source.stem}.o"),
            ],
            check=True,
            capture_output=True,
        )
    command = [
        compiler,
        "-std=c11",
        "-O2",
        *definitions,
        "-Wall",
        "-Wextra",
        "-Wno-unused-parameter",
        "-fsanitize=address,undefined",
        "-fno-sanitize-recover=all",
        str(compiled.artifacts.model_source),
        str(compiled.artifacts.weights_source),
        str(compiled.artifacts.kernels_source),
        *(str(source) for source in support_sources),
        str(runner),
        *include_flags,
        "-o",
        str(executable),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return compiled, executable


def _compare(compiled, executable: Path, *, count: int, seed: int) -> None:  # type: ignore[no-untyped-def]
    input_type = compiled.plan.tensors[compiled.plan.inputs[0]].tensor_type
    rng = np.random.default_rng(seed)
    inputs = rng.integers(
        -128,
        128,
        size=(count, *input_type.shape[1:]),
        dtype=np.int16,
    ).astype(np.int8)
    inputs[:3] = np.asarray(
        [
            np.full(input_type.shape[1:], -128, dtype=np.int8),
            np.full(input_type.shape[1:], 127, dtype=np.int8),
            np.full(
                input_type.shape[1:],
                input_type.qparams.zero_point,
                dtype=np.int8,
            ),
        ]
    )
    expected = np.concatenate(
        [bakenn.run_reference(compiled.plan, row.reshape(input_type.shape)) for row in inputs],
        axis=0,
    )
    result = subprocess.run(
        executable,
        input=inputs.tobytes(),
        capture_output=True,
        check=True,
        env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0"},
    )
    actual = np.frombuffer(result.stdout, dtype=np.int8).reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected)


def _official_s3_scratch_sizes(artifacts, tmp_path: Path) -> tuple[int, int]:  # type: ignore[no-untyped-def]
    compiler = require_compiler("clang")
    source = tmp_path / "scratch_probe.c"
    source.write_text(
        '''#include "esp_nn_esp32s3.h"
#include <stdio.h>

int main(void) {
    const data_dims_t input = { 4, 4, 2, 1 };
    const data_dims_t conv_filter = { 1, 1, 2, 2 };
    const data_dims_t dw_filter = { 3, 3, 2, 1 };
    const data_dims_t output = { 4, 4, 2, 1 };
    const conv_params_t conv = {
        .stride = { 1, 1 }, .padding = { 0, 0 }, .dilation = { 1, 1 }
    };
    const dw_conv_params_t depthwise = {
        .ch_mult = 1, .stride = { 1, 1 }, .padding = { 1, 1 },
        .dilation = { 1, 1 }
    };
    printf("%d %d\\n",
        esp_nn_get_conv_scratch_size_esp32s3(
            &input, &conv_filter, &output, &conv),
        esp_nn_get_depthwise_conv_scratch_size_esp32s3(
            &input, &dw_filter, &output, &depthwise));
    return 0;
}
''',
        encoding="utf-8",
    )
    vendor_root = artifacts.output_dir / "third_party/esp_nn"
    executable = tmp_path / "scratch_probe"
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-O2",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wno-implicit-function-declaration",
            "-Wno-pointer-to-int-cast",
            str(source),
            str(vendor_root / "src/convolution/esp_nn_conv_esp32s3.c"),
            str(vendor_root / "src/convolution/esp_nn_depthwise_conv_s8_esp32s3.c"),
            "-I",
            str(vendor_root / "include"),
            "-I",
            str(vendor_root / "src/common"),
            "-Wl,-dead_strip" if os.uname().sysname == "Darwin" else "-Wl,--gc-sections",
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
    )
    values = subprocess.run(
        [executable], check=True, capture_output=True, text=True
    ).stdout.split()
    return int(values[0]), int(values[1])


@pytest.mark.parametrize("compiler_name", ["gcc", "clang"])
def test_esp32_generic_optimized_conv_and_depthwise_are_byte_exact(
    tmp_path: Path,
    compiler_name: str,
) -> None:
    compiled, executable = _compile_host(
        residual_ds_cnn_graph(), tmp_path / compiler_name, ESP32, compiler_name
    )
    selected = [item.kernel_id for item in compiled.artifacts.backend_plan.selections]
    assert selected[:3] == [
        "esp_nn.esp32.conv2d_s8.v1.2.6",
        "esp_nn.esp32.depthwise_conv2d_s8.v1.2.6",
        "esp_nn.esp32.conv2d_s8.v1.2.6",
    ]
    assert compiled.artifacts.backend_plan.scratch_size == 0
    _compare(
        compiled,
        executable,
        count=10_000 if compiler_name == "gcc" else 256,
        seed=320032,
    )


@pytest.mark.parametrize(
    ("graph", "expected_id"),
    [
        (residual_ds_cnn_graph(), "esp_nn.esp32s3.conv2d_s8.v1.2.6"),
        (linear_graph(32, 16), "esp_nn.esp32s3.linear_per_channel_s8.v1.2.6"),
        (_pool_graph(AveragePool2DOp), "esp_nn.esp32s3.average_pool2d_s8.v1.2.6"),
        (_pool_graph(MaxPool2DOp), "esp_nn.esp32s3.max_pool2d_s8.v1.2.6"),
    ],
)
def test_esp32s3_wrappers_match_reference_through_official_ansi_oracle(
    tmp_path: Path,
    graph: QuantizedGraph,
    expected_id: str,
) -> None:
    compiled, executable = _compile_host(graph, tmp_path / graph.name, ESP32_S3, "clang")
    assert expected_id in {
        item.kernel_id for item in compiled.artifacts.backend_plan.selections
    }
    _compare(compiled, executable, count=10_000, seed=len(graph.ops) * 1203)


def test_esp_nn_fallbacks_are_explicit_and_require_optimized_fails_closed(
    tmp_path: Path,
) -> None:
    average_tie = _pool_graph(AveragePool2DOp, zero_point=37, kernel=(2, 2))
    compiled = bakenn.compile(
        average_tie,
        tmp_path / "average_tie",
        backend_options=_options(ESP32_S3),
        target=ESP32_S3,
    )
    selection = compiled.artifacts.backend_plan.selections[0]
    assert selection.kernel_id == "portable.average_pool2d_s8.v1"
    rejected = selection.rejected[
        "esp_nn.esp32s3.average_pool2d_s8.v1.2.6"
    ]
    assert "centered half-away rounding" in rejected

    small_linear = linear_graph(3, 5)
    with pytest.raises(CompileError, match="no supported implementation"):
        bakenn.compile(
            small_linear,
            tmp_path / "required",
            backend_options=_options(
                ESP32_S3, policy=bakenn.KernelPolicy.REQUIRE_OPTIMIZED
            ),
            target=ESP32_S3,
        )

    no_esp = bakenn.compile(
        residual_ds_cnn_graph(),
        tmp_path / "disabled",
        backend_options=bakenn.CBackendOptions(
            kernel_policy=bakenn.KernelPolicy.AUTO,
            target=ESP32,
        ),
        target=ESP32,
    )
    assert not any(
        item.kernel_id.startswith("esp_nn.")
        for item in no_esp.artifacts.backend_plan.selections
    )
    assert not no_esp.artifacts.support_sources


def test_esp_nn_bundle_and_esp_idf_project_are_pinned_and_self_contained(
    tmp_path: Path,
) -> None:
    compiled = bakenn.compile(
        residual_ds_cnn_graph(),
        tmp_path / "generated",
        backend_options=_options(ESP32_S3),
        target=ESP32_S3,
    )
    artifacts = compiled.artifacts
    assert len(artifacts.support_sources) == 48
    assert len(artifacts.support_include_dirs) == 2
    assert len(artifacts.third_party_licenses) == 1
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    dependency = manifest["bundled_dependencies"][0]
    assert dependency["name"] == "ESP-NN"
    assert dependency["version"] == "1.2.6"
    assert dependency["revision"] == "c0876179f1cf4b4b9073b4f81cb65c8051ccb476"
    assert dependency["target"] == "esp32s3"
    assert dependency["requantization"] == {
        "profile": "TFLM-compatible double rounding",
        "CONFIG_NN_SKIP_NUDGE": False,
    }
    assert "CONFIG_NN_OPTIMIZED=1" in artifacts.build_fragment.read_text(
        encoding="utf-8"
    )

    project = export_esp_idf_project(
        artifacts, ESP32_S3, tmp_path / "esp_idf"
    )
    component_cmake = (project.component / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    assert "CONFIG_NN_OPTIMIZED=1" in component_cmake
    assert "CONFIG_NN_SKIP_NUDGE" not in component_cmake
    assert "esp_nn_conv_esp32s3.c" in component_cmake
    assert "esp_nn_conv_s8_mult8_1x1_esp32s3.S" in component_cmake
    assert (project.component / "third_party/esp_nn/include/esp_nn.h").is_file()
    assert (project.component / "third_party/esp_nn/LICENSE").is_file()
    assert _official_s3_scratch_sizes(artifacts, tmp_path) == (336, 32)


def test_enable_esp_nn_option_requires_boolean() -> None:
    with pytest.raises(ValueError, match="enable_esp_nn"):
        bakenn.CBackendOptions(enable_esp_nn=1)  # type: ignore[arg-type]
