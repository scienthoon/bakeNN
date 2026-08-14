"""Small deterministic fully-quantized P0 whole-model fixtures."""

from __future__ import annotations

import numpy as np

from bakenn.ir import (
    AddOp,
    AveragePool2DOp,
    Conv2DOp,
    DType,
    DepthwiseConv2DOp,
    FlattenOp,
    Layout,
    LinearOp,
    MaxPool2DOp,
    PerAxisQParams,
    PerTensorQParams,
    QuantizedGraph,
    TensorType,
)


def _activation(
    shape: tuple[int, ...],
    qparams: PerTensorQParams,
    layout: Layout,
) -> TensorType:
    return TensorType(shape, DType.INT8, layout, qparams)


def _weight(
    values: np.ndarray,
    scales: tuple[float, ...],
    layout: Layout,
    axis: int,
) -> TensorType:
    return TensorType(
        tuple(values.shape),
        DType.INT8,
        layout,
        PerAxisQParams(scales, (0,) * len(scales), axis),
    )


def _bias(scales: tuple[float, ...]) -> TensorType:
    return TensorType(
        (len(scales),),
        DType.INT32,
        Layout.C,
        PerAxisQParams(scales, (0,) * len(scales), 0),
    )


def _codes(shape: tuple[int, ...], *, offset: int) -> np.ndarray:
    size = int(np.prod(shape))
    values = ((np.arange(size, dtype=np.int16) * 5 + offset) % 19) - 9
    return values.reshape(shape).astype(np.int8)


def tiny_cnn_graph() -> QuantizedGraph:
    """Conv3x3 -> MaxPool -> Flatten(view) -> Linear."""

    input_q = PerTensorQParams(0.25, -3)
    feature_q = PerTensorQParams(0.5, 1)
    output_q = PerTensorQParams(0.25, -5)
    conv_weight_scales = (0.125, 0.25)
    linear_weight_scales = (0.25, 0.125, 0.5)
    conv_weight = _codes((2, 3, 3, 1), offset=1)
    conv_bias = np.asarray([4, -7], dtype=np.int32)
    linear_weight = _codes((3, 8), offset=3)
    linear_bias = np.asarray([5, -3, 9], dtype=np.int32)

    return QuantizedGraph(
        name="p0_tiny_cnn",
        values={
            "input": _activation((1, 4, 4, 1), input_q, Layout.NHWC),
            "conv.weight": _weight(conv_weight, conv_weight_scales, Layout.OHWI, 0),
            "conv.bias": _bias(tuple(input_q.scale * scale for scale in conv_weight_scales)),
            "conv.output": _activation((1, 4, 4, 2), feature_q, Layout.NHWC),
            "pool.output": _activation((1, 2, 2, 2), feature_q, Layout.NHWC),
            "flatten.output": _activation((1, 8), feature_q, Layout.NC),
            "classifier.weight": _weight(linear_weight, linear_weight_scales, Layout.OI, 0),
            "classifier.bias": _bias(
                tuple(feature_q.scale * scale for scale in linear_weight_scales)
            ),
            "output": _activation((1, 3), output_q, Layout.NC),
        },
        constants={
            "conv.weight": conv_weight,
            "conv.bias": conv_bias,
            "classifier.weight": linear_weight,
            "classifier.bias": linear_bias,
        },
        ops=(
            Conv2DOp(
                "conv",
                "input",
                "conv.weight",
                "conv.bias",
                "conv.output",
                padding=(1, 1, 1, 1),
                activation_min=feature_q.zero_point,
            ),
            MaxPool2DOp(
                "pool",
                "conv.output",
                "pool.output",
                kernel=(2, 2),
                stride=(2, 2),
                activation_min=feature_q.zero_point,
            ),
            FlattenOp("flatten", "pool.output", "flatten.output"),
            LinearOp(
                "classifier",
                "flatten.output",
                "classifier.weight",
                "classifier.bias",
                "output",
            ),
        ),
        inputs=("input",),
        outputs=("output",),
    )


def residual_ds_cnn_graph() -> QuantizedGraph:
    """Pointwise stem -> depthwise -> pointwise + long-lived stem residual."""

    input_q = PerTensorQParams(0.25, -2)
    residual_q = PerTensorQParams(0.25, -2)
    depthwise_q = PerTensorQParams(0.5, 3)
    output_q = PerTensorQParams(0.25, -2)
    stem_scales = (0.125, 0.25)
    depthwise_scales = (0.125, 0.25)
    project_scales = (0.25, 0.125)
    stem_weight = _codes((2, 1, 1, 2), offset=2)
    depthwise_weight = _codes((3, 3, 2), offset=4)
    project_weight = _codes((2, 1, 1, 2), offset=7)

    return QuantizedGraph(
        name="p0_residual_ds_cnn",
        values={
            "input": _activation((1, 4, 4, 2), input_q, Layout.NHWC),
            "stem.weight": _weight(stem_weight, stem_scales, Layout.OHWI, 0),
            "stem.bias": _bias(tuple(input_q.scale * scale for scale in stem_scales)),
            "block.residual": _activation((1, 4, 4, 2), residual_q, Layout.NHWC),
            "depthwise.weight": _weight(depthwise_weight, depthwise_scales, Layout.HWO, 2),
            "depthwise.bias": _bias(
                tuple(residual_q.scale * scale for scale in depthwise_scales)
            ),
            "depthwise.output": _activation((1, 4, 4, 2), depthwise_q, Layout.NHWC),
            "project.weight": _weight(project_weight, project_scales, Layout.OHWI, 0),
            "project.bias": _bias(
                tuple(depthwise_q.scale * scale for scale in project_scales)
            ),
            "project.output": _activation((1, 4, 4, 2), output_q, Layout.NHWC),
            "output": _activation((1, 4, 4, 2), output_q, Layout.NHWC),
        },
        constants={
            "stem.weight": stem_weight,
            "stem.bias": np.asarray([2, -3], dtype=np.int32),
            "depthwise.weight": depthwise_weight,
            "depthwise.bias": np.asarray([4, -5], dtype=np.int32),
            "project.weight": project_weight,
            "project.bias": np.asarray([-6, 7], dtype=np.int32),
        },
        ops=(
            Conv2DOp(
                "stem",
                "input",
                "stem.weight",
                "stem.bias",
                "block.residual",
                activation_min=residual_q.zero_point,
            ),
            DepthwiseConv2DOp(
                "depthwise",
                "block.residual",
                "depthwise.weight",
                "depthwise.bias",
                "depthwise.output",
                depth_multiplier=1,
                padding=(1, 1, 1, 1),
                activation_min=depthwise_q.zero_point,
            ),
            Conv2DOp(
                "project",
                "depthwise.output",
                "project.weight",
                "project.bias",
                "project.output",
            ),
            AddOp(
                "residual_add",
                "project.output",
                "block.residual",
                "output",
                activation_min=output_q.zero_point,
            ),
        ),
        inputs=("input",),
        outputs=("output",),
    )


def mobilenet_v1_graph() -> QuantizedGraph:
    """Depthwise -> pointwise -> global AvgPool -> Flatten(view) -> Linear."""

    input_q = PerTensorQParams(0.25, 5)
    feature_q = PerTensorQParams(0.5, -1)
    output_q = PerTensorQParams(0.25, 2)
    depthwise_scales = (0.125, 0.25)
    pointwise_scales = (0.25, 0.125, 0.5)
    linear_scales = (0.25, 0.125)
    depthwise_weight = _codes((3, 3, 2), offset=5)
    pointwise_weight = _codes((3, 1, 1, 2), offset=8)
    linear_weight = _codes((2, 3), offset=11)

    return QuantizedGraph(
        name="p0_mobilenet_v1",
        values={
            "input": _activation((1, 4, 4, 2), input_q, Layout.NHWC),
            "depthwise.weight": _weight(depthwise_weight, depthwise_scales, Layout.HWO, 2),
            "depthwise.bias": _bias(
                tuple(input_q.scale * scale for scale in depthwise_scales)
            ),
            "depthwise.output": _activation((1, 4, 4, 2), feature_q, Layout.NHWC),
            "pointwise.weight": _weight(pointwise_weight, pointwise_scales, Layout.OHWI, 0),
            "pointwise.bias": _bias(
                tuple(feature_q.scale * scale for scale in pointwise_scales)
            ),
            "pointwise.output": _activation((1, 4, 4, 3), feature_q, Layout.NHWC),
            "pool.output": _activation((1, 1, 1, 3), feature_q, Layout.NHWC),
            "flatten.output": _activation((1, 3), feature_q, Layout.NC),
            "classifier.weight": _weight(linear_weight, linear_scales, Layout.OI, 0),
            "classifier.bias": _bias(
                tuple(feature_q.scale * scale for scale in linear_scales)
            ),
            "output": _activation((1, 2), output_q, Layout.NC),
        },
        constants={
            "depthwise.weight": depthwise_weight,
            "depthwise.bias": np.asarray([3, -4], dtype=np.int32),
            "pointwise.weight": pointwise_weight,
            "pointwise.bias": np.asarray([5, -6, 7], dtype=np.int32),
            "classifier.weight": linear_weight,
            "classifier.bias": np.asarray([-8, 9], dtype=np.int32),
        },
        ops=(
            DepthwiseConv2DOp(
                "depthwise",
                "input",
                "depthwise.weight",
                "depthwise.bias",
                "depthwise.output",
                depth_multiplier=1,
                padding=(1, 1, 1, 1),
                activation_min=feature_q.zero_point,
            ),
            Conv2DOp(
                "pointwise",
                "depthwise.output",
                "pointwise.weight",
                "pointwise.bias",
                "pointwise.output",
                activation_min=feature_q.zero_point,
            ),
            AveragePool2DOp(
                "global_pool",
                "pointwise.output",
                "pool.output",
                kernel=(4, 4),
                stride=(4, 4),
                activation_min=feature_q.zero_point,
            ),
            FlattenOp("flatten", "pool.output", "flatten.output"),
            LinearOp(
                "classifier",
                "flatten.output",
                "classifier.weight",
                "classifier.bias",
                "output",
            ),
        ),
        inputs=("input",),
        outputs=("output",),
    )


def representative_graphs() -> tuple[QuantizedGraph, ...]:
    return (tiny_cnn_graph(), residual_ds_cnn_graph(), mobilenet_v1_graph())


__all__ = [
    "mobilenet_v1_graph",
    "representative_graphs",
    "residual_ds_cnn_graph",
    "tiny_cnn_graph",
]
