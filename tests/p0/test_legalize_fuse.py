from __future__ import annotations

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
    verify_graph,
)
from bakenn.ir.ops.conv import Conv2DOp, DepthwiseConv2DOp
from bakenn.ir.ops.elementwise import AddOp, ClampOp, MulOp, RequantizeOp
from bakenn.ir.ops.pool import AveragePool2DOp, MaxPool2DOp
from bakenn.ir.ops.shape import ConcatenateOp
from bakenn.passes import fuse_clamps, legalize_graph


Q = PerTensorQParams(0.25, -3)


def _nc(features: int = 1, qparams: PerTensorQParams = Q) -> TensorType:
    return TensorType((1, features), DType.INT8, Layout.NC, qparams)


def _mixed_concat_graph() -> QuantizedGraph:
    input_qparams = PerTensorQParams(0.125, 7)
    constant_qparams = PerTensorQParams(0.5, -11)
    return QuantizedGraph(
        name="mixed_concat",
        values={
            "input": _nc(2, input_qparams),
            "constant": _nc(1, constant_qparams),
            "aligned": _nc(1, Q),
            "output": _nc(4, Q),
        },
        constants={
            "constant": np.asarray([[17]], dtype=np.int8),
            "aligned": np.asarray([[-8]], dtype=np.int8),
        },
        ops=(ConcatenateOp("concat", ("input", "constant", "aligned"), "output", -1),),
        inputs=("input",),
        outputs=("output",),
    )


def test_legalize_concat_inserts_one_requantize_per_mismatched_edge() -> None:
    source = _mixed_concat_graph()
    first = legalize_graph(source)
    second = legalize_graph(source)

    assert [type(op) for op in first.ops] == [RequantizeOp, RequantizeOp, ConcatenateOp]
    assert [op.name for op in first.ops] == [op.name for op in second.ops]
    assert [op.outputs for op in first.ops] == [op.outputs for op in second.ops]
    assert source.ops[0].inputs == ("input", "constant", "aligned")
    concat = first.ops[-1]
    assert isinstance(concat, ConcatenateOp)
    assert concat.inputs[2] == "aligned"
    assert concat.inputs[0] != "input"
    assert concat.inputs[1] != "constant"
    assert len(set(concat.inputs)) == 3
    for name in concat.inputs:
        assert first.values[name].qparams == first.values["output"].qparams
    assert set(first.constants) == set(source.constants)
    for name in source.constants:
        np.testing.assert_array_equal(first.constants[name], source.constants[name])
        assert first.constants[name] is not source.constants[name]
    verify_graph(first)


def test_legalize_is_idempotent_and_never_inserts_redundant_nodes() -> None:
    once = legalize_graph(_mixed_concat_graph())
    twice = legalize_graph(once)
    assert twice.ops == once.ops
    assert twice.values == once.values
    assert len(twice.ops) == 3

    aligned = QuantizedGraph(
        "aligned_concat",
        {
            "input": _nc(1),
            "constant": _nc(1),
            "output": _nc(2),
        },
        {"constant": np.asarray([[2]], dtype=np.int8)},
        (ConcatenateOp("concat", ("input", "constant"), "output", 1),),
        ("input",),
        ("output",),
    )
    result = legalize_graph(aligned)
    assert result.ops == aligned.ops
    assert not any(isinstance(op, RequantizeOp) for op in result.ops)


def test_public_compile_runs_legalization_before_lowering(tmp_path) -> None:
    compiled = bakenn.compile(_mixed_concat_graph(), tmp_path)
    assert [type(step).__name__ for step in compiled.plan.steps] == [
        "RequantizeStep",
        "RequantizeStep",
        "ConcatenateStep",
    ]


def _linear_graph() -> QuantizedGraph:
    weight_qparams = PerAxisQParams((0.5,), (0,), 0)
    bias_qparams = PerAxisQParams((Q.scale * 0.5,), (0,), 0)
    return QuantizedGraph(
        "linear_clamp",
        {
            "input": _nc(),
            "weight": TensorType((1, 1), DType.INT8, Layout.OI, weight_qparams),
            "bias": TensorType((1,), DType.INT32, Layout.C, bias_qparams),
            "middle": _nc(),
            "output": _nc(),
        },
        {
            "weight": np.asarray([[3]], dtype=np.int8),
            "bias": np.asarray([0], dtype=np.int32),
        },
        (
            LinearOp("producer", "input", "weight", "bias", "middle", -5, 5),
            ClampOp("clamp", "middle", "output", 10, 20),
        ),
        ("input",),
        ("output",),
    )


def _conv_graph(depthwise: bool) -> QuantizedGraph:
    weight_qparams = PerAxisQParams((0.5,), (0,), 2 if depthwise else 0)
    bias_qparams = PerAxisQParams((Q.scale * 0.5,), (0,), 0)
    weight_shape = (1, 1, 1) if depthwise else (1, 1, 1, 1)
    weight_layout = Layout.HWO if depthwise else Layout.OHWI
    values = {
        "input": TensorType((1, 1, 1, 1), DType.INT8, Layout.NHWC, Q),
        "weight": TensorType(weight_shape, DType.INT8, weight_layout, weight_qparams),
        "bias": TensorType((1,), DType.INT32, Layout.C, bias_qparams),
        "middle": TensorType((1, 1, 1, 1), DType.INT8, Layout.NHWC, Q),
        "output": TensorType((1, 1, 1, 1), DType.INT8, Layout.NHWC, Q),
    }
    producer = (
        DepthwiseConv2DOp("producer", "input", "weight", "bias", "middle")
        if depthwise
        else Conv2DOp("producer", "input", "weight", "bias", "middle")
    )
    return QuantizedGraph(
        "depthwise_clamp" if depthwise else "conv_clamp",
        values,
        {
            "weight": np.asarray([[[3]]], dtype=np.int8)
            if depthwise
            else np.asarray([[[[3]]]], dtype=np.int8),
            "bias": np.asarray([0], dtype=np.int32),
        },
        (producer, ClampOp("clamp", "middle", "output", 0, 11)),
        ("input",),
        ("output",),
    )


def _binary_graph(op_type: type[AddOp] | type[MulOp]) -> QuantizedGraph:
    return QuantizedGraph(
        f"{op_type.__name__}_clamp",
        {
            "input": _nc(),
            "constant": _nc(),
            "middle": _nc(),
            "output": _nc(),
        },
        {"constant": np.asarray([[4]], dtype=np.int8)},
        (
            op_type("producer", "input", "constant", "middle", -20, 30),
            ClampOp("clamp", "middle", "output", 0, 10),
        ),
        ("input",),
        ("output",),
    )


def _pool_graph(
    op_type: type[AveragePool2DOp] | type[MaxPool2DOp],
) -> QuantizedGraph:
    tensor = TensorType((1, 1, 1, 1), DType.INT8, Layout.NHWC, Q)
    return QuantizedGraph(
        f"{op_type.__name__}_clamp",
        {"input": tensor, "middle": tensor, "output": tensor},
        {},
        (
            op_type("producer", "input", "middle", (1, 1), (1, 1)),
            ClampOp("clamp", "middle", "output", -2, 12),
        ),
        ("input",),
        ("output",),
    )


@pytest.mark.parametrize(
    "factory",
    [
        _linear_graph,
        lambda: _conv_graph(False),
        lambda: _conv_graph(True),
        lambda: _binary_graph(AddOp),
        lambda: _binary_graph(MulOp),
        lambda: _pool_graph(AveragePool2DOp),
        lambda: _pool_graph(MaxPool2DOp),
    ],
)
def test_fuse_clamp_into_every_eligible_single_consumer_producer(factory) -> None:  # type: ignore[no-untyped-def]
    source = factory()
    verify_graph(source)
    result = fuse_clamps(source)
    assert len(source.ops) == 2
    assert len(result.ops) == 1
    producer = result.ops[0]
    assert producer.output == "output"
    assert "middle" not in result.values
    assert not any(isinstance(op, ClampOp) for op in result.ops)
    assert result.values["output"].qparams == source.values["middle"].qparams
    verify_graph(result)
    assert fuse_clamps(result).ops == result.ops


def test_fuse_composes_disjoint_clamps_as_a_constant_interval() -> None:
    result = fuse_clamps(_linear_graph())
    producer = result.ops[0]
    assert producer.activation_min == 10
    assert producer.activation_max == 10


def test_fuse_preserves_clamp_on_diamond_fanout() -> None:
    graph = QuantizedGraph(
        "diamond",
        {
            "input": _nc(),
            "constant": _nc(),
            "middle": _nc(),
            "clipped": _nc(),
            "output": _nc(),
        },
        {"constant": np.asarray([[1]], dtype=np.int8)},
        (
            AddOp("producer", "input", "constant", "middle"),
            ClampOp("clamp", "middle", "clipped", 0, 127),
            AddOp("join", "middle", "clipped", "output"),
        ),
        ("input",),
        ("output",),
    )
    result = fuse_clamps(graph)
    assert result.ops == graph.ops
    assert "middle" in result.values
    assert isinstance(result.ops[1], ClampOp)


def test_fuse_clamp_into_requantize_without_removing_rounding_point() -> None:
    input_qparams = PerTensorQParams(0.5, 7)
    graph = QuantizedGraph(
        "rounding_point",
        {
            "input": _nc(1, input_qparams),
            "middle": _nc(),
            "output": _nc(),
        },
        {},
        (
            RequantizeOp("requantize", "input", "middle"),
            ClampOp("clamp", "middle", "output", 0, 100),
        ),
        ("input",),
        ("output",),
    )
    result = fuse_clamps(graph)
    assert result.ops == (
        RequantizeOp(
            "requantize",
            "input",
            "output",
            activation_min=0,
            activation_max=100,
        ),
    )
    assert isinstance(result.ops[0], RequantizeOp)
