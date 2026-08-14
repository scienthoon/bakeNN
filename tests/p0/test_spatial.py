from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bakenn.errors import GraphValidationError
from bakenn.ir import (
    ConvTranspose2DOp,
    DType,
    Layout,
    PerAxisQParams,
    PerTensorQParams,
    QuantizedGraph,
    ResizeBilinear2DOp,
    ResizeNearest2DOp,
    TensorType,
    verify_graph,
)
from bakenn.plan import lower_to_plan
from bakenn.reference import run_reference


QPARAMS = PerTensorQParams(0.125, -3)


def _resize_graph(*, bilinear: bool, output_shape: tuple[int, ...]) -> QuantizedGraph:
    op = (
        ResizeBilinear2DOp("resize", "input", "output", align_corners=True)
        if bilinear
        else ResizeNearest2DOp("resize", "input", "output")
    )
    return QuantizedGraph(
        "resize_golden",
        {
            "input": TensorType((1, 2, 2, 1), DType.INT8, Layout.NHWC, QPARAMS),
            "output": TensorType(output_shape, DType.INT8, Layout.NHWC, QPARAMS),
        },
        {},
        (op,),
        ("input",),
        ("output",),
    )


def _transpose_graph() -> QuantizedGraph:
    input_qparams = PerTensorQParams(0.5, 0)
    weight_qparams = PerAxisQParams((0.25,), (0,), 0)
    return QuantizedGraph(
        "transpose_golden",
        {
            "input": TensorType((1, 2, 2, 1), DType.INT8, Layout.NHWC, input_qparams),
            "weight": TensorType((1, 2, 2, 1), DType.INT8, Layout.OHWI, weight_qparams),
            "bias": TensorType(
                (1,),
                DType.INT32,
                Layout.C,
                PerAxisQParams((0.125,), (0,), 0),
            ),
            "output": TensorType(
                (1, 4, 4, 1),
                DType.INT8,
                Layout.NHWC,
                PerTensorQParams(0.125, 0),
            ),
        },
        {
            "weight": np.ones((1, 2, 2, 1), dtype=np.int8),
            "bias": np.zeros((1,), dtype=np.int32),
        },
        (
            ConvTranspose2DOp(
                "transpose",
                "input",
                "weight",
                "bias",
                "output",
                stride=(2, 2),
            ),
        ),
        ("input",),
        ("output",),
    )


def _grouped_transpose_graph() -> QuantizedGraph:
    input_qparams = PerTensorQParams(0.5, 0)
    weight_qparams = PerAxisQParams((0.25, 0.25), (0, 0), 0)
    return QuantizedGraph(
        "grouped_transpose_golden",
        {
            "input": TensorType((1, 1, 1, 2), DType.INT8, Layout.NHWC, input_qparams),
            "weight": TensorType((2, 1, 1, 1), DType.INT8, Layout.OHWI, weight_qparams),
            "bias": TensorType(
                (2,), DType.INT32, Layout.C, PerAxisQParams((0.125, 0.125), (0, 0), 0)
            ),
            "output": TensorType(
                (1, 1, 1, 2), DType.INT8, Layout.NHWC, PerTensorQParams(0.125, 0)
            ),
        },
        {
            "weight": np.asarray([[[[2]]], [[[3]]]], dtype=np.int8),
            "bias": np.zeros((2,), dtype=np.int32),
        },
        (
            ConvTranspose2DOp(
                "grouped_transpose",
                "input",
                "weight",
                "bias",
                "output",
                groups=2,
            ),
        ),
        ("input",),
        ("output",),
    )


def test_resize_nearest_and_bilinear_hand_goldens() -> None:
    nearest = run_reference(
        lower_to_plan(_resize_graph(bilinear=False, output_shape=(1, 4, 4, 1))),
        np.asarray([[[[1], [2]], [[3], [4]]]], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        nearest[..., 0],
        np.asarray([[[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]], dtype=np.int8),
    )

    bilinear = run_reference(
        lower_to_plan(_resize_graph(bilinear=True, output_shape=(1, 3, 3, 1))),
        np.asarray([[[[0], [4]], [[8], [12]]]], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        bilinear[..., 0],
        np.asarray([[[0, 2, 4], [4, 6, 8], [8, 10, 12]]], dtype=np.int8),
    )


def test_conv_transpose_stride_two_hand_golden() -> None:
    actual = run_reference(
        lower_to_plan(_transpose_graph()),
        np.asarray([[[[1], [2]], [[3], [4]]]], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        actual[..., 0],
        np.asarray([[[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]], dtype=np.int8),
    )

    base = _transpose_graph()
    asymmetric = replace(
        base,
        values={
            **base.values,
            "output": replace(base.values["output"], shape=(1, 4, 3, 1)),
        },
        ops=(
            replace(
                base.ops[0],
                padding=(1, 0, 0, 1),
                output_padding=(1, 0),
            ),
        ),
    )
    actual_asymmetric = run_reference(
        lower_to_plan(asymmetric),
        np.asarray([[[[1], [2]], [[3], [4]]]], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        actual_asymmetric[..., 0],
        np.asarray([[[1, 1, 2], [3, 3, 4], [3, 3, 4], [0, 0, 0]]], dtype=np.int8),
    )


def test_grouped_conv_transpose_keeps_group_channels_isolated() -> None:
    plan = lower_to_plan(_grouped_transpose_graph())
    assert plan.steps[0].groups == 2
    actual = run_reference(plan, np.asarray([[[[4, 5]]]], dtype=np.int8))
    np.testing.assert_array_equal(actual, np.asarray([[[[8, 15]]]], dtype=np.int8))


def test_spatial_contracts_fail_closed() -> None:
    resize = _resize_graph(bilinear=False, output_shape=(1, 4, 4, 1))
    with pytest.raises(GraphValidationError, match="preserves per-tensor qparams"):
        verify_graph(
            replace(
                resize,
                values={
                    **resize.values,
                    "output": replace(
                        resize.values["output"],
                        qparams=PerTensorQParams(0.25, -3),
                    ),
                },
            )
        )

    transpose = _transpose_graph()
    with pytest.raises(GraphValidationError, match="output_padding must be smaller"):
        verify_graph(
            replace(
                transpose,
                ops=(replace(transpose.ops[0], output_padding=(2, 0)),),
            )
        )
    with pytest.raises(GraphValidationError, match="output shape"):
        verify_graph(
            replace(
                transpose,
                values={
                    **transpose.values,
                    "output": replace(transpose.values["output"], shape=(1, 5, 4, 1)),
                },
            )
        )
