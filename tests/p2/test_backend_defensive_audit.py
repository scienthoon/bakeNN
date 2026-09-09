"""Regressions for the pinned vendor kernels' narrower arithmetic contracts."""

from dataclasses import replace
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

import bakenn
from bakenn.backend.portable_c import select_backend_plan
from bakenn.errors import CompileError
from bakenn.ir import (
    Conv2DOp, DType, DepthwiseConv2DOp, Layout, LinearOp, PerAxisQParams,
    PerTensorQParams, QuantizedGraph, TensorType,
)
from bakenn.plan import lower_to_plan
from bakenn.targets import CORTEX_M4, ESP32, ESP32_S3

from .support import require_compiler


def _options(target, *, packing=False, policy=bakenn.KernelPolicy.STATIC_PRIORITY):
    return bakenn.CBackendOptions(
        target=target,
        enable_cmsis_nn=target is CORTEX_M4,
        enable_esp_nn=target in (ESP32, ESP32_S3),
        enable_weight_packing=packing,
        kernel_policy=policy,
    )


def _conv_graph(
    *, depthwise=False, height=1, channels=1, kernel_height=1,
    depth_multiplier=1, padding=0,
) -> QuantizedGraph:
    output_channels = channels * depth_multiplier if depthwise else 1
    weight_shape = (
        (kernel_height, 1, output_channels) if depthwise
        else (output_channels, kernel_height, 1, channels)
    )
    weight_q = PerAxisQParams(
        (0.5,) * output_channels, (0,) * output_channels, 2 if depthwise else 0
    )
    q = PerTensorQParams(1.0, 0)
    pad = (padding, padding, padding, padding) if isinstance(padding, int) else padding
    operation = (
        DepthwiseConv2DOp(
            "op", "input", "weight", "bias", "output",
            depth_multiplier=depth_multiplier, padding=pad,
        )
        if depthwise else Conv2DOp("op", "input", "weight", "bias", "output", padding=pad)
    )
    return QuantizedGraph(
        name="vendor_boundary",
        values={
            "input": TensorType((1, height, 1, channels), DType.INT8, Layout.NHWC, q),
            "weight": TensorType(
                weight_shape, DType.INT8, Layout.HWO if depthwise else Layout.OHWI, weight_q,
            ),
            "bias": TensorType(
                (output_channels,), DType.INT32, Layout.C,
                PerAxisQParams((0.5,) * output_channels, (0,) * output_channels, 0),
            ),
            "output": TensorType(
                (1, height + pad[0] + pad[1] - kernel_height + 1, 1 + pad[2] + pad[3], output_channels),
                DType.INT8, Layout.NHWC, q,
            ),
        },
        constants={
            "weight": np.ones(weight_shape, dtype=np.int8),
            "bias": np.zeros(output_channels, dtype=np.int32),
        },
        ops=(operation,), inputs=("input",), outputs=("output",),
    )


def _arithmetic_graph(kind: str, *, negative=False, tiny_scale=False) -> QuantizedGraph:
    # These float32 scales produce multiplier=2147483589, shift=0; unlike an
    # exact real multiplier of one they permit a near-INT32_MAX accumulator.
    input_scale = 0.5 if tiny_scale else 1.9881641864776611
    weight_scale = 2.0 ** -31 if tiny_scale else 1.142663836479187
    output_scale = 1.0 if tiny_scale else 2.271803379058838
    zero_point = 0 if tiny_scale else (-128 if negative else 127)
    bias = 1 if tiny_scale else (-(2 ** 31 - 1) if negative else 2 ** 31 - 1)
    if kind == "linear":
        input_shape, output_shape, weight_shape = (1, 256), (1, 1), (1, 256)
        activation_layout, weight_layout, axis = Layout.NC, Layout.OI, 0
        op_type = LinearOp
    elif kind == "conv":
        input_shape = output_shape = (1, 1, 1, 1)
        weight_shape = (1, 1, 1, 1)
        activation_layout, weight_layout, axis = Layout.NHWC, Layout.OHWI, 0
        op_type = Conv2DOp
    else:
        input_shape = output_shape = (1, 1, 1, 1)
        weight_shape = (1, 1, 1)
        activation_layout, weight_layout, axis = Layout.NHWC, Layout.HWO, 2
        op_type = DepthwiseConv2DOp
    return QuantizedGraph(
        name="arithmetic_boundary",
        values={
            "input": TensorType(input_shape, DType.INT8, activation_layout, PerTensorQParams(input_scale, 0)),
            "weight": TensorType(weight_shape, DType.INT8, weight_layout, PerAxisQParams((weight_scale,), (0,), axis)),
            "bias": TensorType((1,), DType.INT32, Layout.C, PerAxisQParams((input_scale * weight_scale,), (0,), 0)),
            "output": TensorType(output_shape, DType.INT8, activation_layout, PerTensorQParams(output_scale, zero_point)),
        },
        constants={"weight": np.zeros(weight_shape, dtype=np.int8), "bias": np.asarray([bias], dtype=np.int32)},
        ops=(op_type("op", "input", "weight", "bias", "output"),),
        inputs=("input",), outputs=("output",),
    )


def _run_c(graph, target, tmp_path: Path, inputs: np.ndarray) -> tuple[object, np.ndarray]:
    compiled = bakenn.compile(
        graph, tmp_path / "model", model_name="audit", target=target,
        backend_options=_options(target),
    )
    artifacts = compiled.artifacts
    executable = tmp_path / "runner"
    # Exact-size objects ensure sanitizer diagnostics catch reads as well as
    # writes outside the tensor; input/output bytes are compared to a golden.
    runner = f'''#include "{artifacts.header.name}"
#include <stdio.h>
static int8_t input[BKNN_AUDIT_INPUT_SIZE];
static int8_t output[BKNN_AUDIT_OUTPUT_SIZE];
static _Alignas(BKNN_AUDIT_ARENA_ALIGNMENT)
    uint8_t arena[BKNN_AUDIT_ARENA_SIZE + 1u];
int main(void) {{
    if (fread(input, 1, sizeof(input), stdin) != sizeof(input)) return 2;
    bknn_audit_infer(arena, input, output);
    return fwrite(output, 1, sizeof(output), stdout) != sizeof(output);
}}
'''
    command = [
        require_compiler(os.environ.get("CC", "cc")), "-std=c11", "-O2", "-fsanitize=address,undefined",
        "-fno-sanitize-recover=all", "-D__GNUC_PYTHON__", "-D__RESTRICT=restrict",
        "-DBAKENN_CMSIS_NN_BUILTIN_MEMORY", "-DCONFIG_NN_OPTIMIZED=1",
        "-I", str(artifacts.output_dir),
        *(flag for directory in artifacts.support_include_dirs for flag in ("-I", str(directory))),
        str(artifacts.model_source), str(artifacts.weights_source), str(artifacts.kernels_source),
        *(str(source) for source in artifacts.support_sources),
        "-x", "c", "-", "-o", str(executable),
    ]
    result = subprocess.run(command, input=runner.encode(), capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    result = subprocess.run(
        [str(executable)], input=inputs.tobytes(), capture_output=True,
        env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0"},
    )
    assert result.returncode == 0, result.stderr.decode()
    return compiled, np.frombuffer(result.stdout, dtype=np.int8)


@pytest.mark.parametrize("padding", [1, 2])
def test_esp32_padded_1x1_uses_safe_fallback(tmp_path: Path, padding: int) -> None:
    graph = _conv_graph(padding=padding)
    compiled, actual = _run_c(graph, ESP32, tmp_path, np.asarray([2], dtype=np.int8))
    expected = np.zeros((1 + 2 * padding, 1 + 2 * padding), dtype=np.int8)
    expected[padding, padding] = 1
    np.testing.assert_array_equal(actual, expected.ravel())
    assert "ignores padding" in compiled.artifacts.backend_plan.selections[0].rejected["esp_nn.esp32.conv2d_s8.v1.2.6"]


@pytest.mark.parametrize("height", [32768, 32769])
def test_cmsis_depthwise_int16_origin_boundary(tmp_path: Path, height: int) -> None:
    graph = _conv_graph(depthwise=True, height=height, depth_multiplier=2)
    compiled, actual = _run_c(graph, CORTEX_M4, tmp_path, np.full(height, 2, dtype=np.int8))
    np.testing.assert_array_equal(actual, np.ones(height * 2, dtype=np.int8))
    assert compiled.artifacts.backend_plan.selections[0].kernel_id.startswith("cmsis_nn.") == (height == 32768)


@pytest.mark.parametrize("channels", [32767, 32768])
def test_cmsis_conv_uint16_reduction_boundary(tmp_path: Path, channels: int) -> None:
    graph = _conv_graph(height=3, channels=channels, kernel_height=2)
    compiled, actual = _run_c(graph, CORTEX_M4, tmp_path, np.full(3 * channels, 2, dtype=np.int8))
    np.testing.assert_array_equal(actual, np.asarray([127, 127], dtype=np.int8))
    assert compiled.artifacts.backend_plan.selections[0].kernel_id.startswith("cmsis_nn.") == (channels == 32767)


@pytest.mark.parametrize("padding", [1, 2])
def test_cmsis_depthwise_padding_respects_dsp_scratch(tmp_path: Path, padding: int) -> None:
    graph = _conv_graph(depthwise=True, padding=padding)
    compiled, actual = _run_c(graph, CORTEX_M4, tmp_path, np.asarray([2], dtype=np.int8))
    expected = np.zeros((1 + 2 * padding, 1 + 2 * padding), dtype=np.int8)
    expected[padding, padding] = 1
    np.testing.assert_array_equal(actual, expected.ravel())
    selected = compiled.artifacts.backend_plan.selections[0]
    assert selected.kernel_id.startswith("cmsis_nn.") == (padding == 1)
    if padding == 2:
        assert "DSP im2col" in selected.rejected["cmsis_nn.depthwise_conv2d_s8.v4.0.0"]


@pytest.mark.parametrize("height", [32768, 32769])
@pytest.mark.parametrize("depth_multiplier", [1, 2])
def test_esp32_depthwise_int16_origin_boundary(tmp_path: Path, height: int, depth_multiplier: int) -> None:
    graph = _conv_graph(depthwise=True, height=height, depth_multiplier=depth_multiplier)
    compiled, actual = _run_c(graph, ESP32, tmp_path, np.full(height, 2, dtype=np.int8))
    np.testing.assert_array_equal(actual, np.ones(height * depth_multiplier, dtype=np.int8))
    selected = compiled.artifacts.backend_plan.selections[0]
    assert selected.kernel_id.startswith("esp_nn.") == (height == 32768)
    if height == 32769:
        assert "int16" in selected.rejected["esp_nn.esp32.depthwise_conv2d_s8.v1.2.6"]


@pytest.mark.parametrize("depth_multiplier,channels,uses_generic", [
    (1, 1, True), (2, 1, True), (4, 1, False), (1, 8, False),
])
def test_s3_dispatcher_applies_int16_guard_to_generic_branch_only(
    depth_multiplier: int, channels: int, uses_generic: bool,
) -> None:
    graph = _conv_graph(depthwise=True, height=32769, channels=channels,
                        depth_multiplier=depth_multiplier)
    selected = select_backend_plan(lower_to_plan(graph), _options(ESP32_S3)).selections[0]
    assert selected.kernel_id.startswith("esp_nn.") == (not uses_generic)
    if uses_generic:
        assert "int16" in selected.rejected["esp_nn.esp32s3.depthwise_conv2d_s8.v1.2.6"]


@pytest.mark.parametrize("padding,depth_multiplier", [((0, 0, 2, 2), 1), ((2, 2, 2, 2), 2)])
def test_cmsis_safe_horizontal_padding_and_generic_branch_remain_supported(
    tmp_path: Path, padding: tuple[int, ...], depth_multiplier: int,
) -> None:
    graph = _conv_graph(depthwise=True, padding=padding, depth_multiplier=depth_multiplier)
    compiled, actual = _run_c(graph, CORTEX_M4, tmp_path, np.asarray([2], dtype=np.int8))
    expected = np.zeros(graph.values["output"].shape, dtype=np.int8)
    expected[0, padding[0], padding[2], :] = 1
    np.testing.assert_array_equal(actual, expected.ravel())
    assert compiled.artifacts.backend_plan.selections[0].kernel_id.startswith("cmsis_nn.")


def test_require_optimized_rejects_cmsis_dsp_scratch_overflow() -> None:
    plan = lower_to_plan(_conv_graph(depthwise=True, padding=2))
    with pytest.raises(CompileError, match="DSP im2col"):
        select_backend_plan(plan, _options(CORTEX_M4, policy=bakenn.KernelPolicy.REQUIRE_OPTIMIZED))


@pytest.mark.parametrize("target,kind", [(CORTEX_M4, "linear"), (CORTEX_M4, "conv"), (CORTEX_M4, "depthwise"), (ESP32, "conv"), (ESP32, "depthwise")])
@pytest.mark.parametrize("negative", [False, True])
def test_vendor_output_offset_overflow_falls_back(tmp_path: Path, target, kind: str, negative: bool) -> None:
    graph = _arithmetic_graph(kind, negative=negative)
    compiled, actual = _run_c(graph, target, tmp_path, np.zeros(graph.values["input"].shape, dtype=np.int8))
    np.testing.assert_array_equal(actual, np.asarray([-128 if negative else 127], dtype=np.int8))
    assert "overflow int32" in " ".join(compiled.artifacts.backend_plan.selections[0].rejected.values())


@pytest.mark.parametrize("target,kind", [(CORTEX_M4, "linear"), (CORTEX_M4, "conv"), (CORTEX_M4, "depthwise"), (ESP32, "conv"), (ESP32, "depthwise")])
def test_vendor_signed_shift_mask_boundary_falls_back(tmp_path: Path, target, kind: str) -> None:
    graph = _arithmetic_graph(kind, tiny_scale=True)
    assert lower_to_plan(graph).steps[0].shifts == (-31,)
    compiled, actual = _run_c(graph, target, tmp_path, np.zeros(graph.values["input"].shape, dtype=np.int8))
    np.testing.assert_array_equal(actual, np.asarray([0], dtype=np.int8))
    assert "signed shift masks" in " ".join(compiled.artifacts.backend_plan.selections[0].rejected.values())


@pytest.mark.parametrize("kind", ["linear", "conv", "depthwise"])
@pytest.mark.parametrize("tiny_scale", [False, True])
def test_esp32s3_arithmetic_constraints_fail_closed(kind: str, tiny_scale: bool) -> None:
    plan = lower_to_plan(_arithmetic_graph(kind, tiny_scale=tiny_scale))
    selection = select_backend_plan(plan, _options(ESP32_S3, packing=True)).selections[0]
    reason = selection.rejected[next(key for key in selection.rejected if key.startswith("esp_nn."))]
    assert ("signed shift masks" if tiny_scale else "overflow int32") in reason


def test_require_optimized_rejects_unsafe_vendor_candidate() -> None:
    plan = lower_to_plan(_conv_graph(padding=1))
    with pytest.raises(CompileError, match="ignores padding"):
        select_backend_plan(plan, _options(ESP32, policy=bakenn.KernelPolicy.REQUIRE_OPTIMIZED))


@pytest.mark.parametrize("target", [CORTEX_M4, ESP32, ESP32_S3])
def test_supported_conv_still_uses_vendor_kernel(target) -> None:
    plan = lower_to_plan(_conv_graph())
    selected = select_backend_plan(plan, _options(target)).selections[0]
    assert selected.kernel_id.startswith("cmsis_nn." if target is CORTEX_M4 else "esp_nn.")
