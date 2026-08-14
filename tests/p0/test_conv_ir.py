from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bakenn.errors import CompileError, GraphValidationError
from bakenn.ir.graph import QuantizedGraph
from bakenn.ir.ops.conv import Conv2DOp, DepthwiseConv2DOp
from bakenn.ir.types import DType, Layout, PerAxisQParams, PerTensorQParams, TensorType
from bakenn.ir.verify import verify_graph
from bakenn.plan.lower import lower_to_plan
from bakenn.plan.steps.conv import Conv2DStep, DepthwiseConv2DStep
from bakenn.quantization.fixedpoint import INT32_MAX
from bakenn.reference.executor import run_reference

# Explicit imports install this work package's registrations.  Central
# aggregators are owned by WP-50.
import bakenn.ir.verifiers.conv  # noqa: F401
import bakenn.plan.lowering.conv  # noqa: F401
import bakenn.reference.kernels.conv  # noqa: F401


def _conv_graph(
    *,
    input_shape: tuple[int, int, int, int] = (1, 2, 3, 1),
    output_shape: tuple[int, int, int, int] = (1, 2, 3, 2),
    padding: tuple[int, int, int, int] = (1, 0, 1, 0),
    stride: tuple[int, int] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
    input_zero_point: int = -5,
    activation_min: int = -128,
    activation_max: int = 127,
) -> QuantizedGraph:
    input_qparams = PerTensorQParams(0.25, input_zero_point)
    output_qparams = PerTensorQParams(0.5, 3)
    weight_qparams = PerAxisQParams((0.1, 0.2), (0, 0), 0)
    bias_qparams = PerAxisQParams((0.025, 0.05), (0, 0), 0)
    return QuantizedGraph(
        name="conv_golden",
        values={
            "input": TensorType(input_shape, DType.INT8, Layout.NHWC, input_qparams),
            "weight": TensorType((2, 2, 2, 1), DType.INT8, Layout.OHWI, weight_qparams),
            "bias": TensorType((2,), DType.INT32, Layout.C, bias_qparams),
            "output": TensorType(output_shape, DType.INT8, Layout.NHWC, output_qparams),
        },
        constants={
            "weight": np.asarray(
                [
                    [[[-1], [2]], [[3], [-4]]],
                    [[[5], [-6]], [[7], [8]]],
                ],
                dtype=np.int8,
            ),
            "bias": np.asarray([11, -13], dtype=np.int32),
        },
        ops=(
            Conv2DOp(
                "conv",
                "input",
                "weight",
                "bias",
                "output",
                stride=stride,
                dilation=dilation,
                padding=padding,
                activation_min=activation_min,
                activation_max=activation_max,
            ),
        ),
        inputs=("input",),
        outputs=("output",),
    )


def _depthwise_graph() -> QuantizedGraph:
    input_qparams = PerTensorQParams(0.125, 17)
    output_qparams = PerTensorQParams(0.25, -9)
    scales = (0.1, 0.2, 0.3, 0.4)
    return QuantizedGraph(
        name="depthwise_golden",
        values={
            "input": TensorType((1, 3, 3, 2), DType.INT8, Layout.NHWC, input_qparams),
            "weight": TensorType(
                (2, 2, 4), DType.INT8, Layout.HWO, PerAxisQParams(scales, (0,) * 4, 2)
            ),
            "bias": TensorType(
                (4,),
                DType.INT32,
                Layout.C,
                PerAxisQParams(tuple(0.125 * scale for scale in scales), (0,) * 4, 0),
            ),
            "output": TensorType((1, 2, 2, 4), DType.INT8, Layout.NHWC, output_qparams),
        },
        constants={
            "weight": np.asarray(
                [
                    [[1, -2, 3, -4], [5, -6, 7, -8]],
                    [[9, -10, 11, -12], [13, -14, 15, -16]],
                ],
                dtype=np.int8,
            ),
            "bias": np.asarray([3, -4, 5, -6], dtype=np.int32),
        },
        ops=(
            DepthwiseConv2DOp(
                "depthwise",
                "input",
                "weight",
                "bias",
                "output",
                depth_multiplier=2,
            ),
        ),
        inputs=("input",),
        outputs=("output",),
    )


def test_conv_hand_golden_uses_asymmetric_padding_zero_point_and_clamp() -> None:
    graph = _conv_graph(activation_min=-2, activation_max=5)
    plan = lower_to_plan(graph)
    assert isinstance(plan.steps[0], Conv2DStep)
    input_values = np.asarray([[[[-10], [-2], [6]], [[12], [20], [-30]]]], dtype=np.int8)
    actual = run_reference(plan, input_values)
    expected = np.asarray(
        [[[[5, -2], [2, 1], [2, 5]], [[0, 5], [2, 5], [5, -2]]]], dtype=np.int8
    )
    np.testing.assert_array_equal(actual, expected)

    # With an all-zero-point input, padded and in-bounds positions both have
    # centered value zero.  Only quantized bias remains.
    centered_zero = np.full(graph.values["input"].shape, -5, dtype=np.int8)
    bias_only = run_reference(plan, centered_zero)
    assert np.unique(bias_only[..., 0]).size == 1
    assert np.unique(bias_only[..., 1]).size == 1


def test_depthwise_hand_golden_depth_multiplier_mapping() -> None:
    plan = lower_to_plan(_depthwise_graph())
    assert isinstance(plan.steps[0], DepthwiseConv2DStep)
    input_values = np.asarray(
        [
            [
                [[1, 2], [3, 4], [5, 6]],
                [[7, 8], [9, 10], [11, 12]],
                [[13, 14], [15, 16], [17, 18]],
            ]
        ],
        dtype=np.int8,
    )
    actual = run_reference(plan, input_values)
    expected = np.asarray(
        [[[[-23, 23, -59, 67], [-20, 17, -49, 51]],
          [[-14, 4, -27, 19], [-12, -2, -16, 3]]]],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda graph: replace(graph, ops=(replace(graph.ops[0], groups=2),)), "groups"),
        (lambda graph: replace(graph, ops=(replace(graph.ops[0], stride=(0, 1)),)), "positive"),
        (lambda graph: replace(graph, ops=(replace(graph.ops[0], dilation=(1, 0)),)), "positive"),
        (lambda graph: replace(graph, ops=(replace(graph.ops[0], padding=(-1, 0, 0, 0)),)), "nonnegative"),
        (
            lambda graph: replace(
                graph,
                values={**graph.values, "output": replace(graph.values["output"], shape=(1, 9, 9, 2))},
            ),
            "output shape",
        ),
        (
            lambda graph: replace(
                graph,
                values={
                    **graph.values,
                    "weight": replace(
                        graph.values["weight"],
                        qparams=PerAxisQParams((0.1, 0.2), (1, 0), 0),
                    ),
                },
            ),
            "zero_point zero",
        ),
    ],
)
def test_conv_rejects_malformed_contracts(mutation, message: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(GraphValidationError, match=message):
        verify_graph(mutation(_conv_graph()))


def test_depthwise_rejects_wrong_multiplier_and_axis() -> None:
    graph = _depthwise_graph()
    with pytest.raises(GraphValidationError, match="C_out"):
        verify_graph(replace(graph, ops=(replace(graph.ops[0], depth_multiplier=1),)))
    invalid_values = dict(graph.values)
    invalid_values["weight"] = replace(
        graph.values["weight"],
        qparams=PerAxisQParams((0.1, 0.2), (0, 0), 1),
    )
    with pytest.raises(GraphValidationError, match="axis 2"):
        verify_graph(replace(graph, values=invalid_values))


def test_conv_accumulator_and_positive_shift_proofs_fail_closed() -> None:
    accumulator_graph = _conv_graph()
    accumulator_constants = dict(accumulator_graph.constants)
    accumulator_constants["bias"] = np.asarray([INT32_MAX, 0], dtype=np.int32)
    with pytest.raises(CompileError, match="accumulator bound"):
        lower_to_plan(replace(accumulator_graph, constants=accumulator_constants))

    shift_graph = _conv_graph()
    shift_values = dict(shift_graph.values)
    shift_values["output"] = replace(
        shift_graph.values["output"], qparams=PerTensorQParams(1e-8, 0)
    )
    with pytest.raises(CompileError, match="left shift"):
        lower_to_plan(replace(shift_graph, values=shift_values))


def test_conv_stride_dilation_and_asymmetric_shape_contract() -> None:
    graph = _conv_graph(
        input_shape=(1, 5, 6, 1),
        output_shape=(1, 3, 3, 2),
        padding=(2, 1, 1, 0),
        stride=(2, 2),
        dilation=(2, 2),
    )
    verify_graph(graph)
    result = run_reference(lower_to_plan(graph), np.zeros((1, 5, 6, 1), dtype=np.int8))
    assert result.shape == (1, 3, 3, 2)
