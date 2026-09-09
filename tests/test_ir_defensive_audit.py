"""Boundary regressions for malformed typed IR that previously reached C emission."""

from dataclasses import replace

import numpy as np
import pytest

import bakenn
from bakenn.errors import GraphValidationError
from bakenn.ir import (
    AveragePool1DOp, Conv1DOp, ConvTranspose2DOp, DType, Layout,
    MaxPool1DOp, PerAxisQParams, PerTensorQParams, QuantizedGraph,
    ReduceMeanOp, SliceOp, TensorType, verify_graph,
)


def _unary(op, shape, output_shape, layout):  # type: ignore[no-untyped-def]
    qparams = PerTensorQParams(1.0, 0)
    return QuantizedGraph(
        "audit", {
            "input": TensorType(shape, DType.INT8, layout, qparams),
            "output": TensorType(output_shape, DType.INT8, layout, qparams),
        }, {}, (op,), ("input",), ("output",),
    )


@pytest.mark.parametrize(
    ("layout", "shape", "output_shape", "axes"),
    [
        (Layout.NHWC, (1, 2, 3), (1, 1, 1), (1, 2)),
        (Layout.NHWC, (1, 2, 3, 2, 2), (1, 1, 1, 2, 2), (1, 2)),
        (Layout.NLC, (1, 2, 3, 2), (1, 1, 3, 2), (1,)),
    ],
)
def test_reduce_mean_rejects_layout_rank_mismatch_before_emission(
    layout, shape, output_shape, axes, tmp_path  # type: ignore[no-untyped-def]
) -> None:
    graph = _unary(ReduceMeanOp("mean", "input", "output", axes), shape, output_shape, layout)
    with pytest.raises(GraphValidationError, match="rank"):
        bakenn.compile(graph, tmp_path / "generated")
    assert not (tmp_path / "generated").exists()


@pytest.mark.parametrize("axes", [(5, 6), (-7, -6), (1, 6)])
def test_reduce_mean_rejects_out_of_rank_axes(axes) -> None:  # type: ignore[no-untyped-def]
    graph = _unary(
        ReduceMeanOp("mean", "input", "output", axes),
        (1, 2, 3, 1), (1, 1, 1, 1), Layout.NHWC,
    )
    with pytest.raises(GraphValidationError, match="axes.*range"):
        verify_graph(graph)


def test_reduce_mean_accepts_negative_axes_in_range() -> None:
    graph = _unary(
        ReduceMeanOp("mean", "input", "output", (-3, -2)),
        (1, 2, 3, 1), (1, 1, 1, 1), Layout.NHWC,
    )
    verify_graph(graph)


@pytest.mark.parametrize("op_type", [AveragePool1DOp, MaxPool1DOp])
@pytest.mark.parametrize(
    "kwargs", [{"kernel": 1, "stride": 1 << 32},
               {"kernel": (1 << 32) + 1, "stride": 1, "padding": (1 << 32, 0)}],
)
def test_pool1d_rejects_parameters_outside_target_abi(op_type, kwargs) -> None:  # type: ignore[no-untyped-def]
    graph = _unary(op_type("pool", "input", "output", **kwargs), (1, 1, 1), (1, 1, 1), Layout.NLC)
    with pytest.raises(GraphValidationError, match="target ABI"):
        verify_graph(graph)


def _compute_graph(op_type):  # type: ignore[no-untyped-def]
    sequence = op_type is Conv1DOp
    shape = (1, 1, 1) if sequence else (1, 1, 1, 1)
    layout = Layout.NLC if sequence else Layout.NHWC
    weight_layout = Layout.OWI if sequence else Layout.OHWI
    qparams = PerTensorQParams(1.0, 0)
    channel_qparams = PerAxisQParams((1.0,), (0,), 0)
    return QuantizedGraph(
        "compute_audit", {
            "input": TensorType(shape, DType.INT8, layout, qparams),
            "output": TensorType(shape, DType.INT8, layout, qparams),
            "weight": TensorType(shape, DType.INT8, weight_layout, channel_qparams),
            "bias": TensorType((1,), DType.INT32, Layout.C, channel_qparams),
        }, {"weight": np.ones(shape, dtype=np.int8), "bias": np.zeros((1,), dtype=np.int32)},
        (op_type("conv", "input", "weight", "bias", "output"),), ("input",), ("output",),
    )


@pytest.mark.parametrize("field", ["stride", "dilation"])
def test_transpose_rejects_parameters_outside_target_abi(field: str) -> None:
    graph = _compute_graph(ConvTranspose2DOp)
    graph = replace(graph, ops=(replace(graph.ops[0], **{field: (1 << 32, 1)}),))
    with pytest.raises(GraphValidationError, match="target ABI"):
        verify_graph(graph)


@pytest.mark.parametrize("op_type", [Conv1DOp, ConvTranspose2DOp])
@pytest.mark.parametrize("scale", [1e-30, 1e30])
def test_extended_convolution_reports_unrepresentable_bias_scale(op_type, scale) -> None:  # type: ignore[no-untyped-def]
    graph = _compute_graph(op_type)
    values = dict(graph.values)
    values["input"] = replace(values["input"], qparams=PerTensorQParams(scale, 0))
    values["weight"] = replace(values["weight"], qparams=PerAxisQParams((scale,), (0,), 0))
    graph = replace(graph, values=values)
    with pytest.raises(GraphValidationError, match="bias scale.*float32"):
        verify_graph(graph)


def test_slice_rejects_per_axis_activation_qparams() -> None:
    graph = _unary(
        SliceOp("slice", "input", "output", 1, 0, 1),
        (1, 2, 2), (1, 1, 2), Layout.NLC,
    )
    qparams = PerAxisQParams((1.0, 2.0), (0, 0), 2)
    graph = replace(graph, values={name: replace(value, qparams=qparams) for name, value in graph.values.items()})
    with pytest.raises(GraphValidationError, match="per-tensor"):
        verify_graph(graph)
