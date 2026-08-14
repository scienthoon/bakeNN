from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import bakenn

from bakenn.errors import CompileError
from bakenn.ir import (
    Conv2DOp,
    DType,
    DepthwiseConv2DOp,
    Layout,
    LinearOp,
    PerAxisQParams,
    PerTensorQParams,
    QuantizedGraph,
    TensorType,
    verify_graph,
)
from bakenn.passes import analyze_constant_channels, deduplicate_constants
from bakenn.plan import lower_to_plan
from bakenn.reference import run_reference


def _constant_channel_graph(kind: str) -> QuantizedGraph:
    input_qparams = PerTensorQParams(0.25, 5)
    weight_qparams = PerAxisQParams((0.5, 0.25), (0, 0), 0 if kind != "dw" else 2)
    bias_qparams = PerAxisQParams(
        tuple(input_qparams.scale * scale for scale in weight_qparams.scales),
        (0, 0),
        0,
    )
    output_qparams = PerTensorQParams(0.125, -2)
    bias = np.asarray([3, 0], dtype=np.int32)

    if kind == "linear":
        input_type = TensorType((1, 3), DType.INT8, Layout.NC, input_qparams)
        weight_type = TensorType((2, 3), DType.INT8, Layout.OI, weight_qparams)
        output_type = TensorType((1, 2), DType.INT8, Layout.NC, output_qparams)
        weight = np.asarray([[0, 0, 0], [1, 0, -1]], dtype=np.int8)
        op = LinearOp("compute", "input", "weight", "bias", "output")
    elif kind == "conv":
        input_type = TensorType((1, 2, 2, 1), DType.INT8, Layout.NHWC, input_qparams)
        weight_type = TensorType((2, 1, 1, 1), DType.INT8, Layout.OHWI, weight_qparams)
        output_type = TensorType((1, 2, 2, 2), DType.INT8, Layout.NHWC, output_qparams)
        weight = np.asarray([[[[0]]], [[[1]]]], dtype=np.int8)
        op = Conv2DOp("compute", "input", "weight", "bias", "output")
    elif kind == "dw":
        input_type = TensorType((1, 2, 2, 1), DType.INT8, Layout.NHWC, input_qparams)
        weight_type = TensorType((1, 1, 2), DType.INT8, Layout.HWO, weight_qparams)
        output_type = TensorType((1, 2, 2, 2), DType.INT8, Layout.NHWC, output_qparams)
        weight = np.asarray([[[0, 1]]], dtype=np.int8)
        op = DepthwiseConv2DOp(
            "compute", "input", "weight", "bias", "output", depth_multiplier=2
        )
    else:
        raise AssertionError(f"unknown test kind: {kind}")

    return QuantizedGraph(
        name=f"constant_{kind}",
        values={
            "input": input_type,
            "weight": weight_type,
            "bias": TensorType((2,), DType.INT32, Layout.C, bias_qparams),
            "output": output_type,
        },
        constants={"weight": weight, "bias": bias},
        ops=(op,),
        inputs=("input",),
        outputs=("output",),
    )


@pytest.mark.parametrize("kind", ["linear", "conv", "dw"])
def test_zero_weight_nonzero_bias_channel_has_exact_constant_code(kind: str) -> None:
    graph = _constant_channel_graph(kind)
    analysis = analyze_constant_channels(graph)

    assert len(analysis) == 1
    channel = analysis[0]
    assert (channel.op_name, channel.op_kind, channel.output, channel.channel) == (
        "compute",
        {"linear": "LinearOp", "conv": "Conv2DOp", "dw": "DepthwiseConv2DOp"}[kind],
        "output",
        0,
    )
    # input_scale * weight_scale / output_scale == 1, so bias code 3 is
    # preserved exactly before adding output zero point -2.
    assert (channel.bias_code, channel.multiplier, channel.shift, channel.output_code) == (
        3,
        1 << 30,
        1,
        1,
    )

    plan = lower_to_plan(graph)
    rng = np.random.default_rng(20260814)
    for _ in range(32):
        sample = rng.integers(
            -128, 128, size=graph.values["input"].shape, dtype=np.int16
        ).astype(np.int8)
        output = run_reference(plan, sample)
        if kind == "linear":
            np.testing.assert_array_equal(output[:, 0], np.asarray([1], dtype=np.int8))
        else:
            np.testing.assert_array_equal(
                output[..., 0], np.ones(output.shape[:-1], dtype=np.int8)
            )


def test_constant_channel_analysis_applies_fused_clamp_and_fails_on_unsafe_shift() -> None:
    graph = _constant_channel_graph("linear")
    clamped = replace(
        graph,
        ops=(replace(graph.ops[0], activation_min=-1, activation_max=0),),
    )
    assert analyze_constant_channels(clamped)[0].output_code == 0

    input_qparams = PerTensorQParams(1.0, 0)
    weight_qparams = PerAxisQParams((float(1 << 29), 0.25), (0, 0), 0)
    bias_qparams = PerAxisQParams((float(1 << 29), 0.25), (0, 0), 0)
    values = dict(graph.values)
    values["input"] = replace(values["input"], qparams=input_qparams)
    values["weight"] = replace(values["weight"], qparams=weight_qparams)
    values["bias"] = replace(values["bias"], qparams=bias_qparams)
    values["output"] = replace(values["output"], qparams=PerTensorQParams(1.0, 0))
    unsafe = replace(
        graph,
        values=values,
        constants={
            "weight": graph.constants["weight"],
            "bias": np.asarray([2, 0], dtype=np.int32),
        },
    )
    verify_graph(unsafe)
    with pytest.raises(CompileError, match="constant-channel.*left shift.*not int32-safe"):
        analyze_constant_channels(unsafe)
    with pytest.raises(CompileError, match="left shift is not int32-safe"):
        lower_to_plan(unsafe)

    minimum_bias = replace(
        graph,
        constants={
            "weight": graph.constants["weight"],
            "bias": np.asarray([-(1 << 31), 0], dtype=np.int32),
        },
    )
    verify_graph(minimum_bias)
    with pytest.raises(CompileError, match="accumulator bound exceeds int32"):
        analyze_constant_channels(minimum_bias)


def _duplicate_constant_graph(*, reverse_constants: bool = False) -> QuantizedGraph:
    activation = PerTensorQParams(0.25, 0)
    weight_qparams = PerAxisQParams((0.5, 0.25), (0, 0), 0)
    bias_qparams = PerAxisQParams((0.125, 0.0625), (0, 0), 0)
    weight_type = TensorType((2, 2), DType.INT8, Layout.OI, weight_qparams)
    bias_type = TensorType((2,), DType.INT32, Layout.C, bias_qparams)
    weight = np.asarray([[0, 0], [1, -1]], dtype=np.int8)
    bias = np.asarray([3, 0], dtype=np.int32)
    constants = {
        "z_weight": weight,
        "z_bias": bias,
        "a_weight": weight.copy(),
        "a_bias": bias.copy(),
    }
    if reverse_constants:
        constants = dict(reversed(tuple(constants.items())))
    return QuantizedGraph(
        name="duplicate_constants",
        values={
            "input": TensorType((1, 2), DType.INT8, Layout.NC, activation),
            "z_weight": weight_type,
            "z_bias": bias_type,
            "a_weight": weight_type,
            "a_bias": bias_type,
            "middle": TensorType((1, 2), DType.INT8, Layout.NC, activation),
            "output": TensorType((1, 2), DType.INT8, Layout.NC, activation),
        },
        constants=constants,
        ops=(
            LinearOp("first", "input", "z_weight", "z_bias", "middle"),
            LinearOp("second", "middle", "a_weight", "a_bias", "output"),
        ),
        inputs=("input",),
        outputs=("output",),
    )


def test_deduplicate_constants_is_typed_immutable_and_order_deterministic() -> None:
    first = deduplicate_constants(_duplicate_constant_graph())
    reversed_input = deduplicate_constants(_duplicate_constant_graph(reverse_constants=True))

    assert tuple(first.constants) == ("a_bias", "a_weight")
    assert tuple(reversed_input.constants) == tuple(first.constants)
    for graph in (first, reversed_input):
        assert graph.ops[0].weight == graph.ops[1].weight == "a_weight"
        assert graph.ops[0].bias == graph.ops[1].bias == "a_bias"
        assert "z_weight" not in graph.values and "z_bias" not in graph.values
        verify_graph(graph)
        with pytest.raises(ValueError):
            graph.constants["a_weight"][0, 0] = 99

    sample = np.asarray([[7, -4]], dtype=np.int8)
    before = run_reference(lower_to_plan(_duplicate_constant_graph()), sample)
    after = run_reference(lower_to_plan(first), sample)
    np.testing.assert_array_equal(after, before)
    assert analyze_constant_channels(first) == analyze_constant_channels(reversed_input)


def test_deduplicate_constants_does_not_cross_quantization_domains() -> None:
    graph = _duplicate_constant_graph()
    values = dict(graph.values)
    values["z_weight"] = replace(
        values["z_weight"],
        qparams=PerAxisQParams((0.75, 0.25), (0, 0), 0),
    )
    values["z_bias"] = replace(
        values["z_bias"],
        qparams=PerAxisQParams((0.1875, 0.0625), (0, 0), 0),
    )
    distinct_domains = replace(graph, values=values)
    verify_graph(distinct_domains)

    result = deduplicate_constants(distinct_domains)
    assert tuple(result.constants) == ("a_bias", "a_weight", "z_bias", "z_weight")
    assert result.ops[0].weight == "z_weight"
    assert result.ops[1].weight == "a_weight"


def test_public_compiler_applies_typed_constant_deduplication(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = _duplicate_constant_graph()
    compiled = bakenn.compile(source, tmp_path / "deduplicated")
    assert tuple(compiled.plan.constants) == ("a_bias", "a_weight")
    assert compiled.plan.steps[0].weight == compiled.plan.steps[1].weight == "a_weight"
    assert compiled.plan.steps[0].bias == compiled.plan.steps[1].bias == "a_bias"


def test_constant_passes_reject_non_graph_inputs() -> None:
    with pytest.raises(CompileError, match="requires a QuantizedGraph"):
        analyze_constant_channels(object())  # type: ignore[arg-type]
    with pytest.raises(CompileError, match="requires a QuantizedGraph"):
        deduplicate_constants(object())  # type: ignore[arg-type]
