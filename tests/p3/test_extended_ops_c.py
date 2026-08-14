from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

import bakenn
from bakenn.ir import (
    AveragePool1DOp,
    Conv1DOp,
    Conv2DOp,
    ClampOp,
    DType,
    HardSwishOp,
    Layout,
    MaxPool1DOp,
    Pad2DOp,
    PerAxisQParams,
    PerTensorQParams,
    QuantizedGraph,
    ReduceMeanOp,
    RequantizeOp,
    SiLUOp,
    SigmoidOp,
    TensorType,
)
from tests.p0.test_models_c import _compile_runner


def _weight(shape: tuple[int, ...], scales: tuple[float, ...], layout: Layout) -> TensorType:
    return TensorType(shape, DType.INT8, layout, PerAxisQParams(scales, (0,) * len(scales), 0))


def _bias(input_q: PerTensorQParams, scales: tuple[float, ...]) -> TensorType:
    normalized = PerAxisQParams(scales, (0,) * len(scales), 0).scales
    return TensorType(
        (len(scales),),
        DType.INT32,
        Layout.C,
        PerAxisQParams(tuple(input_q.scale * scale for scale in normalized), (0,) * len(scales), 0),
    )


def _activation_graph() -> QuantizedGraph:
    input_q = PerTensorQParams(0.125, -5)
    sigmoid_q = PerTensorQParams(1.0 / 256.0, -128)
    output_q = PerTensorQParams(0.01, -10)
    shape = (1, 3, 5, 2)
    return QuantizedGraph(
        "p3_activation_luts",
        {
            "input": TensorType(shape, DType.INT8, Layout.NHWC, input_q),
            "sigmoid": TensorType(shape, DType.INT8, Layout.NHWC, sigmoid_q),
            "hard": TensorType(shape, DType.INT8, Layout.NHWC, output_q),
            "output": TensorType(shape, DType.INT8, Layout.NHWC, output_q),
        },
        {},
        (
            SigmoidOp("sigmoid", "input", "sigmoid"),
            HardSwishOp("hardswish", "sigmoid", "hard"),
            SiLUOp("silu", "hard", "output"),
        ),
        ("input",),
        ("output",),
    )


def _pad_mean_graph() -> QuantizedGraph:
    input_q = PerTensorQParams(0.2, 17)
    output_q = PerTensorQParams(0.1, -3)
    return QuantizedGraph(
        "p3_pad_mean",
        {
            "input": TensorType((1, 2, 3, 3), DType.INT8, Layout.NHWC, input_q),
            "padded": TensorType((1, 5, 6, 3), DType.INT8, Layout.NHWC, input_q),
            "output": TensorType((1, 1, 1, 3), DType.INT8, Layout.NHWC, output_q),
        },
        {},
        (
            Pad2DOp("pad", "input", "padded", (1, 2, 1, 2)),
            ReduceMeanOp("mean", "padded", "output", (1, 2), True),
        ),
        ("input",),
        ("output",),
    )


def _grouped_conv_graph() -> QuantizedGraph:
    input_q = PerTensorQParams(0.25, -9)
    output_q = PerTensorQParams(0.2, 4)
    scales = (0.02, 0.03, 0.04, 0.05)
    weight = ((np.arange(4 * 2 * 2 * 2, dtype=np.int16) * 11) % 31 - 15).reshape(4, 2, 2, 2).astype(np.int8)
    bias = np.asarray([31, -17, 9, -43], dtype=np.int32)
    return QuantizedGraph(
        "p3_grouped_conv",
        {
            "input": TensorType((1, 3, 3, 4), DType.INT8, Layout.NHWC, input_q),
            "weight": _weight(weight.shape, scales, Layout.OHWI),
            "bias": _bias(input_q, scales),
            "output": TensorType((1, 3, 3, 4), DType.INT8, Layout.NHWC, output_q),
        },
        {"weight": weight, "bias": bias},
        (
            Conv2DOp(
                "grouped",
                "input",
                "weight",
                "bias",
                "output",
                padding=(1, 0, 0, 1),
                groups=2,
            ),
        ),
        ("input",),
        ("output",),
    )


def _audio_graph() -> QuantizedGraph:
    input_q = PerTensorQParams(0.1, -4)
    conv_q = PerTensorQParams(0.2, 3)
    output_q = PerTensorQParams(0.15, -7)
    scales = (0.03, 0.04, 0.05, 0.06)
    weight = ((np.arange(24, dtype=np.int16) * 5) % 19 - 9).reshape(4, 3, 2).astype(np.int8)
    bias = np.asarray([1, -2, 3, -4], dtype=np.int32)
    return QuantizedGraph(
        "p3_audio",
        {
            "input": TensorType((1, 8, 4), DType.INT8, Layout.NLC, input_q),
            "weight": _weight(weight.shape, scales, Layout.OWI),
            "bias": _bias(input_q, scales),
            "conv": TensorType((1, 8, 4), DType.INT8, Layout.NLC, conv_q),
            "avg": TensorType((1, 4, 4), DType.INT8, Layout.NLC, conv_q),
            "max": TensorType((1, 2, 4), DType.INT8, Layout.NLC, conv_q),
            "output": TensorType((1, 1, 4), DType.INT8, Layout.NLC, output_q),
        },
        {"weight": weight, "bias": bias},
        (
            Conv1DOp("conv", "input", "weight", "bias", "conv", padding=(1, 1), groups=2),
            AveragePool1DOp("avg", "conv", "avg", 2, 2),
            MaxPool1DOp("max", "avg", "max", 2, 2),
            ReduceMeanOp("mean", "max", "output", (1,), True),
        ),
        ("input",),
        ("output",),
    )


@pytest.mark.parametrize(
    ("factory", "seed"),
    ((_activation_graph, 1), (_pad_mean_graph, 2), (_grouped_conv_graph, 3), (_audio_graph, 4)),
)
def test_extended_ops_python_and_generated_c_are_byte_exact(
    factory, seed: int, tmp_path: Path  # type: ignore[no-untyped-def]
) -> None:
    compiler = shutil.which("clang") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("host C compiler is unavailable")
    graph = factory()
    compiled = bakenn.compile(graph, tmp_path / "generated")
    executable = _compile_runner(compiled, compiled.artifacts.output_dir, compiler)
    input_type = graph.values[graph.inputs[0]]
    rng = np.random.default_rng(seed)
    edge = np.stack(
        (
            np.full(input_type.shape[1:], -128, dtype=np.int8),
            np.full(input_type.shape[1:], 127, dtype=np.int8),
            np.full(input_type.shape[1:], input_type.qparams.zero_point, dtype=np.int8),
        )
    )
    random = rng.integers(-128, 128, size=(64, *input_type.shape[1:]), dtype=np.int16).astype(np.int8)
    inputs = np.concatenate((edge, random))
    expected = np.concatenate(
        [bakenn.run_reference(compiled.plan, sample.reshape(input_type.shape)).reshape(-1) for sample in inputs]
    )
    process = subprocess.run(executable, input=inputs.tobytes(), capture_output=True, check=True)
    actual = np.frombuffer(process.stdout, dtype=np.int8)
    np.testing.assert_array_equal(actual, expected)


def test_requantize_clamp_is_one_loop_after_typed_fusion() -> None:
    input_q = PerTensorQParams(0.5, 7)
    output_q = PerTensorQParams(0.25, -3)
    graph = QuantizedGraph(
        "p3_requant_clamp",
        {
            "input": TensorType((1, 16), DType.INT8, Layout.NC, input_q),
            "middle": TensorType((1, 16), DType.INT8, Layout.NC, output_q),
            "output": TensorType((1, 16), DType.INT8, Layout.NC, output_q),
        },
        {},
        (
            RequantizeOp("requant", "input", "middle"),
            ClampOp("clamp", "middle", "output", -10, 20),
        ),
        ("input",),
        ("output",),
    )
    from bakenn.passes import fuse_clamps

    fused = fuse_clamps(graph)
    assert len(fused.ops) == 1
    assert fused.ops[0].activation_min == -10
    assert fused.ops[0].activation_max == 20
