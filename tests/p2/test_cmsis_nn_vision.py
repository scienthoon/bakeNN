from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

import bakenn
from bakenn.ir import (
    AveragePool2DOp,
    Conv2DOp,
    DType,
    DepthwiseConv2DOp,
    Layout,
    MaxPool2DOp,
    PerAxisQParams,
    PerTensorQParams,
    QuantizedGraph,
    TensorType,
)
from bakenn.targets import CORTEX_M4, PORTABLE_32, build_freestanding_elf
from tests.p0.model_fixtures import (
    mobilenet_v1_graph,
    residual_ds_cnn_graph,
    tiny_cnn_graph,
)

from .support import require_compiler


def _options(*, target=CORTEX_M4, policy=bakenn.KernelPolicy.AUTO):  # type: ignore[no-untyped-def]
    return bakenn.CBackendOptions(
        kernel_policy=policy,
        enable_cmsis_nn=True,
        target=target,
    )


def _compile_host(
    graph: QuantizedGraph,
    output_dir: Path,
    compiler_name: str,
):  # type: ignore[no-untyped-def]
    compiler = require_compiler(compiler_name)
    compiled = bakenn.compile(
        graph,
        output_dir,
        model_name=f"cmsis_{graph.name}",
        backend_options=_options(),
        target=CORTEX_M4,
    )
    artifacts = compiled.artifacts
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    symbol = str(manifest["model"])
    macro = symbol.upper()
    runner = output_dir / "runner.c"
    runner.write_text(
        f'''#include "{artifacts.header.name}"
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#define GUARD_SIZE 32u

static int guard_ok(const uint8_t *guard) {{
    for (size_t index = 0; index < GUARD_SIZE; ++index) {{
        if (guard[index] != UINT8_C(0xA5)) return 0;
    }}
    return 1;
}}

int main(void) {{
    _Alignas({macro}_ARENA_ALIGNMENT)
        uint8_t arena_storage[GUARD_SIZE + {macro}_ARENA_SIZE + GUARD_SIZE];
    struct {{
        uint8_t before[GUARD_SIZE];
        int8_t data[{macro}_INPUT_SIZE];
        uint8_t after[GUARD_SIZE];
    }} input;
    struct {{
        uint8_t before[GUARD_SIZE];
        int8_t data[{macro}_OUTPUT_SIZE];
        uint8_t after[GUARD_SIZE];
    }} output;
    memset(arena_storage, 0xA5, sizeof(arena_storage));
    memset(&input, 0xA5, sizeof(input));
    memset(&output, 0xA5, sizeof(output));
    uint8_t *arena = {macro}_ARENA_SIZE == 0u
        ? NULL : arena_storage + GUARD_SIZE;
    while (fread(input.data, 1u, {macro}_INPUT_SIZE, stdin)
            == {macro}_INPUT_SIZE) {{
        {symbol}_infer(arena, input.data, output.data);
        if (!guard_ok(arena_storage)
            || !guard_ok(arena_storage + GUARD_SIZE + {macro}_ARENA_SIZE)
            || !guard_ok(input.before) || !guard_ok(input.after)
            || !guard_ok(output.before) || !guard_ok(output.after)) return 4;
        if (fwrite(output.data, 1u, {macro}_OUTPUT_SIZE, stdout)
                != {macro}_OUTPUT_SIZE) return 2;
    }}
    return ferror(stdin) ? 3 : 0;
}}
''',
        encoding="utf-8",
    )
    executable = output_dir / "runner"
    command = [
        compiler,
        "-std=c11",
        "-O2",
        "-D__GNUC_PYTHON__",
        "-D__RESTRICT=restrict",
        "-DBAKENN_CMSIS_NN_BUILTIN_MEMORY",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-pedantic",
        "-fsanitize=address,undefined",
        "-fno-sanitize-recover=all",
        str(artifacts.model_source),
        str(artifacts.weights_source),
        str(artifacts.kernels_source),
        *(str(path) for path in artifacts.support_sources),
        str(runner),
        "-I",
        str(artifacts.output_dir),
        *(
            flag
            for include_dir in artifacts.support_include_dirs
            for flag in ("-I", str(include_dir))
        ),
        "-o",
        str(executable),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return compiled, executable


def _run_and_compare(
    compiled,  # type: ignore[no-untyped-def]
    executable: Path,
    *,
    count: int,
    seed: int,
) -> None:
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
    )
    actual = np.frombuffer(result.stdout, dtype=np.int8).reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("compiler_name", ["gcc", "clang"])
@pytest.mark.parametrize(
    ("factory", "required_ids"),
    [
        (
            tiny_cnn_graph,
            {
                "cmsis_nn.conv2d_s8.v4.0.0",
                "cmsis_nn.max_pool2d_s8.v4.0.0",
            },
        ),
        (
            residual_ds_cnn_graph,
            {
                "cmsis_nn.conv2d_s8.v4.0.0",
                "cmsis_nn.depthwise_conv2d_s8.v4.0.0",
            },
        ),
        (
            mobilenet_v1_graph,
            {
                "cmsis_nn.conv2d_s8.v4.0.0",
                "cmsis_nn.depthwise_conv2d_s8.v4.0.0",
            },
        ),
    ],
)
def test_cmsis_nn_cnn_graphs_are_byte_exact_with_sanitizers(
    tmp_path: Path,
    compiler_name: str,
    factory,  # type: ignore[no-untyped-def]
    required_ids: set[str],
) -> None:
    graph = factory()
    compiled, executable = _compile_host(
        graph, tmp_path / f"{compiler_name}_{graph.name}", compiler_name
    )
    selected = {
        selection.kernel_id for selection in compiled.artifacts.backend_plan.selections
    }
    assert required_ids <= selected
    _run_and_compare(compiled, executable, count=256, seed=len(graph.ops) * 101)


def _depth_multiplier_graph() -> QuantizedGraph:
    input_q = PerTensorQParams(0.25, -17)
    output_q = PerTensorQParams(0.5, 9)
    weight_scales = (0.125, 0.25, 0.375, 0.5)
    weight = ((np.arange(36, dtype=np.int16) * 7) % 17 - 8).reshape(3, 3, 4).astype(
        np.int8
    )
    bias = np.asarray((5, -7, 11, -13), dtype=np.int32)
    return QuantizedGraph(
        name="cmsis_depth_multiplier",
        values={
            "input": TensorType((1, 5, 5, 2), DType.INT8, Layout.NHWC, input_q),
            "weight": TensorType(
                (3, 3, 4),
                DType.INT8,
                Layout.HWO,
                PerAxisQParams(weight_scales, (0,) * 4, 2),
            ),
            "bias": TensorType(
                (4,),
                DType.INT32,
                Layout.C,
                PerAxisQParams(
                    tuple(input_q.scale * scale for scale in weight_scales),
                    (0,) * 4,
                    0,
                ),
            ),
            "output": TensorType((1, 3, 3, 4), DType.INT8, Layout.NHWC, output_q),
        },
        constants={"weight": weight, "bias": bias},
        ops=(
            DepthwiseConv2DOp(
                "depthwise",
                "input",
                "weight",
                "bias",
                "output",
                depth_multiplier=2,
                stride=(2, 2),
                padding=(1, 1, 1, 1),
                activation_min=-64,
                activation_max=96,
            ),
        ),
        inputs=("input",),
        outputs=("output",),
    )


def _positive_shift_conv_graph() -> QuantizedGraph:
    input_q = PerTensorQParams(0.5, -31)
    output_q = PerTensorQParams(0.03125, 17)
    weight_scales = (0.5, 0.25)
    weight = np.asarray((3, -2, -4, 5), dtype=np.int8).reshape(2, 1, 1, 2)
    bias = np.asarray((7, -9), dtype=np.int32)
    return QuantizedGraph(
        name="cmsis_positive_shift_conv",
        values={
            "input": TensorType((1, 3, 4, 2), DType.INT8, Layout.NHWC, input_q),
            "weight": TensorType(
                (2, 1, 1, 2),
                DType.INT8,
                Layout.OHWI,
                PerAxisQParams(weight_scales, (0, 0), 0),
            ),
            "bias": TensorType(
                (2,),
                DType.INT32,
                Layout.C,
                PerAxisQParams(
                    tuple(input_q.scale * scale for scale in weight_scales),
                    (0, 0),
                    0,
                ),
            ),
            "output": TensorType((1, 3, 4, 2), DType.INT8, Layout.NHWC, output_q),
        },
        constants={"weight": weight, "bias": bias},
        ops=(
            Conv2DOp(
                "conv",
                "input",
                "weight",
                "bias",
                "output",
                activation_min=-93,
                activation_max=101,
            ),
        ),
        inputs=("input",),
        outputs=("output",),
    )


def _pool_graph(op_type):  # type: ignore[no-untyped-def]
    qparams = PerTensorQParams(
        0.125,
        0 if op_type is AveragePool2DOp else 37,
    )
    return QuantizedGraph(
        name=f"cmsis_{op_type.__name__}",
        values={
            "input": TensorType((1, 4, 5, 3), DType.INT8, Layout.NHWC, qparams),
            "output": TensorType((1, 4, 5, 3), DType.INT8, Layout.NHWC, qparams),
        },
        constants={},
        ops=(
            op_type(
                "pool",
                "input",
                "output",
                kernel=(3, 3),
                stride=(1, 1),
                padding=(1, 1, 1, 1),
                activation_min=-71,
                activation_max=103,
            ),
        ),
        inputs=("input",),
        outputs=("output",),
    )


@pytest.mark.parametrize(
    ("graph", "kernel_id", "scratch_size"),
    [
        (_depth_multiplier_graph(), "cmsis_nn.depthwise_conv2d_s8.v4.0.0", 0),
        (_positive_shift_conv_graph(), "cmsis_nn.conv2d_s8.v4.0.0", 0),
        (_pool_graph(AveragePool2DOp), "cmsis_nn.average_pool2d_s8.v4.0.0", 12),
        (_pool_graph(MaxPool2DOp), "cmsis_nn.max_pool2d_s8.v4.0.0", 0),
    ],
)
def test_cmsis_nn_extended_shapes_are_byte_exact(
    tmp_path: Path,
    graph: QuantizedGraph,
    kernel_id: str,
    scratch_size: int,
) -> None:
    compiled, executable = _compile_host(graph, tmp_path / graph.name, "clang")
    selection = compiled.artifacts.backend_plan.selections[0]
    assert selection.kernel_id == kernel_id
    assert selection.scratch_size == scratch_size
    backend_plan = compiled.artifacts.backend_plan
    assert backend_plan.scratch_size == scratch_size
    if scratch_size:
        assert backend_plan.scratch_offset is not None
        assert backend_plan.scratch_offset >= backend_plan.activation_arena_size
        assert backend_plan.scratch_offset % backend_plan.scratch_alignment == 0
        assert backend_plan.arena_size >= backend_plan.scratch_offset + scratch_size
        assert backend_plan.arena_size - (backend_plan.scratch_offset + scratch_size) < (
            backend_plan.arena_alignment
        )
    else:
        assert backend_plan.scratch_offset is None
    manifest = json.loads(compiled.artifacts.manifest.read_text(encoding="utf-8"))
    assert manifest["scratch_bytes"] == scratch_size
    assert manifest["arena_bytes"] == backend_plan.arena_size
    if graph.name == "cmsis_positive_shift_conv":
        assert any(shift > 0 for shift in compiled.plan.steps[0].shifts)
    _run_and_compare(compiled, executable, count=512, seed=447626)


def test_cmsis_nn_vision_falls_back_when_contract_is_not_supported(
    tmp_path: Path,
) -> None:
    graph = tiny_cnn_graph()
    portable_target = bakenn.compile(
        graph,
        tmp_path / "portable_target",
        backend_options=_options(target=PORTABLE_32),
        target=PORTABLE_32,
    )
    selections = portable_target.artifacts.backend_plan.selections
    assert all(not item.kernel_id.startswith("cmsis_nn.") for item in selections)
    assert not portable_target.artifacts.support_sources

    asymmetric = _pool_graph(MaxPool2DOp)
    op = asymmetric.ops[0]
    asymmetric = QuantizedGraph(
        name="asymmetric_pool",
        values={
            "input": asymmetric.values["input"],
            "output": TensorType(
                (1, 3, 4, 3),
                DType.INT8,
                Layout.NHWC,
                asymmetric.values["output"].qparams,
            ),
        },
        constants={},
        ops=(
            MaxPool2DOp(
                op.name,
                op.input,
                op.output,
                kernel=(3, 3),
                stride=(1, 1),
                padding=(0, 1, 0, 1),
            ),
        ),
        inputs=("input",),
        outputs=("output",),
    )
    fallback = bakenn.compile(
        asymmetric,
        tmp_path / "asymmetric",
        backend_options=_options(),
        target=CORTEX_M4,
    )
    selection = fallback.artifacts.backend_plan.selections[0]
    assert selection.kernel_id == "portable.max_pool2d_s8.v1"
    assert "symmetric padding" in selection.rejected[
        "cmsis_nn.max_pool2d_s8.v4.0.0"
    ]

    nonzero_qparams = PerTensorQParams(0.125, 37)
    even_window_average = QuantizedGraph(
        name="average_rounding_fallback",
        values={
            "input": TensorType(
                (1, 4, 4, 1), DType.INT8, Layout.NHWC, nonzero_qparams
            ),
            "output": TensorType(
                (1, 2, 2, 1), DType.INT8, Layout.NHWC, nonzero_qparams
            ),
        },
        constants={},
        ops=(
            AveragePool2DOp(
                "average",
                "input",
                "output",
                kernel=(2, 2),
                stride=(2, 2),
            ),
        ),
        inputs=("input",),
        outputs=("output",),
    )
    rounding_fallback = bakenn.compile(
        even_window_average,
        tmp_path / "average_rounding",
        backend_options=_options(),
        target=CORTEX_M4,
    )
    selection = rounding_fallback.artifacts.backend_plan.selections[0]
    assert selection.kernel_id != "cmsis_nn.average_pool2d_s8.v4.0.0"
    assert "half-away rounding" in selection.rejected[
        "cmsis_nn.average_pool2d_s8.v4.0.0"
    ]


def test_cmsis_nn_vision_bundle_cross_links_without_runtime_or_heap(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("arm-none-eabi-gcc")
    if compiler is None:
        if os.environ.get("BAKENN_REQUIRE_ARM_CC") == "1":
            pytest.fail("ARM cross compiler is required by CI")
        pytest.skip("ARM cross compiler is unavailable")
    cases = (
        (
            tiny_cnn_graph(),
            ("arm_convolve_wrapper_s8.c", "arm_max_pool_s8.c"),
        ),
        (
            mobilenet_v1_graph(),
            ("arm_convolve_wrapper_s8.c", "arm_depthwise_conv_wrapper_s8.c"),
        ),
        (
            _pool_graph(AveragePool2DOp),
            ("arm_avgpool_s8.c",),
        ),
    )
    for graph, expected_sources in cases:
        compiled = bakenn.compile(
            graph,
            tmp_path / graph.name,
            backend_options=_options(),
            target=CORTEX_M4,
        )
        report = build_freestanding_elf(
            compiled.artifacts,
            CORTEX_M4,
            tmp_path / f"{graph.name}_elf",
            compiler=compiler,
        )
        assert report.undefined_symbols == ()
        assert report.forbidden_symbols == ()
        manifest = json.loads(compiled.artifacts.manifest.read_text(encoding="utf-8"))
        sources = manifest["bundled_dependencies"][0]["sources"]
        for expected_source in expected_sources:
            assert any(expected_source in value for value in sources)
