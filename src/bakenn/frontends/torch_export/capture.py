from __future__ import annotations

from dataclasses import replace
import math
from numbers import Integral, Real
from typing import Any

import numpy as np

from bakenn.errors import CompileError
from .model import (
    FloatAddOp,
    FloatAveragePool1DOp,
    FloatAveragePool2DOp,
    FloatConcatOp,
    FloatConv1DOp,
    FloatConv2DOp,
    FloatConvTranspose2DOp,
    FloatDType,
    FloatDepthwiseConv2DOp,
    FloatFlattenOp,
    FloatGraph,
    FloatHardSigmoidOp,
    FloatHardSwishOp,
    FloatLayout,
    FloatLinearOp,
    FloatMaxPool2DOp,
    FloatMaxPool1DOp,
    FloatMulOp,
    FloatPad2DOp,
    FloatReLU6Op,
    FloatReLUOp,
    FloatReshapeOp,
    FloatReduceMeanOp,
    FloatResizeBilinear2DOp,
    FloatResizeNearest2DOp,
    FloatSigmoidOp,
    FloatSiLUOp,
    FloatSliceOp,
    FloatSoftmaxOp,
    FloatValue,
    FloatValueKind,
)


ALLOWED_ATEN_TARGETS = (
    "aten.add_.Tensor",
    "aten.add.Tensor",
    "aten.adaptive_avg_pool2d.default",
    "aten.avg_pool2d.default",
    "aten.avg_pool1d.default",
    "aten.batch_norm.default",
    "aten.cat.default",
    "aten.conv2d.default",
    "aten.conv_transpose2d.input",
    "aten.conv1d.default",
    "aten.dropout.default",
    "aten.dropout_.default",
    "aten.flatten.using_ints",
    "aten.hardtanh.default",
    "aten.hardtanh_.default",
    "aten.hardswish.default",
    "aten.hardswish_.default",
    "aten.hardsigmoid.default",
    "aten.hardsigmoid_.default",
    "aten.linear.default",
    "aten.max_pool2d.default",
    "aten.max_pool1d.default",
    "aten.mean.dim",
    "aten.mul.Tensor",
    "aten.relu.default",
    "aten.relu_.default",
    "aten.relu6.default",
    "aten.reshape.default",
    "aten.sigmoid.default",
    "aten.silu.default",
    "aten.silu_.default",
    "aten.softmax.int",
    "aten.slice.Tensor",
    "aten.squeeze.dim",
    "aten.unsqueeze.default",
    "aten.upsample_bilinear2d.vec",
    "aten.upsample_nearest2d.vec",
    "aten.pad.default",
    "aten.view.default",
)


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise CompileError(
            "PyTorch torch.export frontend requested, but torch is not installed; "
            "install bakenn[torch]"
        ) from error
    return torch


def _pair(value: object, description: str, *, minimum: int) -> tuple[int, int]:
    if isinstance(value, bool):
        raise CompileError(f"{description} must be an integer or pair")
    if isinstance(value, Integral):
        items = (int(value), int(value))
    else:
        try:
            raw = tuple(value)  # type: ignore[arg-type]
        except TypeError as error:
            raise CompileError(f"{description} must be an integer or pair") from error
        if len(raw) == 0:
            raise CompileError(f"{description} cannot be empty")
        if len(raw) == 1:
            raw = (raw[0], raw[0])
        if len(raw) != 2 or any(
            isinstance(item, bool) or not isinstance(item, Integral) for item in raw
        ):
            raise CompileError(f"{description} must contain two integers")
        items = int(raw[0]), int(raw[1])
    if any(item < minimum for item in items):
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise CompileError(f"{description} must be {qualifier}")
    return items


def _explicit_padding(value: object, description: str) -> tuple[int, int, int, int]:
    height, width = _pair(value, description, minimum=0)
    return height, height, width, width


def _literal_int(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise CompileError(f"{description} must be a static integer")
    return int(value)


def _literal_1d(value: object, description: str, *, minimum: int) -> int:
    if isinstance(value, Integral) and not isinstance(value, bool):
        result = int(value)
    else:
        try:
            items = tuple(value)  # type: ignore[arg-type]
        except TypeError as error:
            raise CompileError(f"{description} must be one static integer") from error
        if len(items) != 1 or isinstance(items[0], bool) or not isinstance(items[0], Integral):
            raise CompileError(f"{description} must be one static integer")
        result = int(items[0])
    if result < minimum:
        raise CompileError(f"{description} must be >= {minimum}")
    return result


def _literal_bool(value: object, description: str) -> bool:
    if not isinstance(value, bool):
        raise CompileError(f"{description} must be a boolean")
    return value


def _layout(shape: tuple[int, ...]) -> FloatLayout:
    if len(shape) == 4:
        return FloatLayout.NCHW
    if len(shape) == 2:
        return FloatLayout.NC
    if len(shape) == 3:
        return FloatLayout.NCL
    if len(shape) == 1:
        return FloatLayout.C
    raise CompileError(f"P0 does not support rank-{len(shape)} FP32 tensors")


def _node_shape_dtype(torch: Any, node: Any) -> tuple[tuple[int, ...], FloatDType]:
    value = node.meta.get("val")
    if value is None or not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise CompileError(f"torch.export node {node.name} has no tensor metadata")
    dimensions: list[int] = []
    for dimension in value.shape:
        # SymInt is deliberately rejected even if it has a concrete hint.
        if type(dimension) is not int or dimension <= 0:
            raise CompileError(f"{node.name}: dynamic or non-positive tensor dimensions are unsupported")
        dimensions.append(dimension)
    if value.dtype is not torch.float32:
        raise CompileError(f"{node.name}: P0 requires float32 tensors, got {value.dtype}")
    shape = tuple(dimensions)
    _layout(shape)
    return shape, FloatDType.FLOAT32


def _activation_value(torch: Any, node: Any) -> FloatValue:
    shape, dtype = _node_shape_dtype(torch, node)
    if len(shape) not in (2, 3, 4) or shape[0] != 1:
        raise CompileError(
            f"{node.name}: activations must be static batch-one rank-two, rank-three, or rank-four tensors"
        )
    return FloatValue(node.name, shape, dtype, _layout(shape), FloatValueKind.ACTIVATION)


def _constant_array(torch: Any, value: object, name: str) -> np.ndarray:
    if not isinstance(value, torch.Tensor):
        raise CompileError(f"{name}: lifted parameter/get_attr value must be a tensor")
    if value.dtype is not torch.float32:
        raise CompileError(f"{name}: P0 constants must be float32, got {value.dtype}")
    array = value.detach().cpu().contiguous().numpy().astype(np.float32, copy=True)
    if array.ndim not in (1, 2, 3, 4) or any(dimension <= 0 for dimension in array.shape):
        raise CompileError(f"{name}: constants must be static rank-one through rank-four tensors")
    if not np.all(np.isfinite(array)):
        raise CompileError(f"{name}: constant contains NaN or infinity")
    return array


def _fetch_attr(root: object, target: str) -> object:
    value = root
    for atom in target.split("."):
        if not atom or not hasattr(value, atom):
            raise CompileError(f"torch.export get_attr target is missing: {target}")
        value = getattr(value, atom)
    return value


def _tensor_name(value: object, values: dict[str, FloatValue], description: str) -> str:
    name = getattr(value, "name", None)
    if not isinstance(name, str) or name not in values:
        raise CompileError(
            f"{description} must be a tensor value; scalar/broadcast operands are unsupported"
        )
    # An eval Dropout key deliberately points at its canonical input value.
    # Returning FloatValue.name resolves that identity edge for all extractors
    # without adding a target operation or mutating the generated input ABI.
    return values[name].name


def _arg(node: Any, index: int, default: object) -> object:
    return node.args[index] if index < len(node.args) else default


def _set_layout(values: dict[str, FloatValue], name: str, layout: FloatLayout) -> None:
    values[name] = replace(values[name], layout=layout)


def _require_constant(constants: dict[str, np.ndarray], name: str, op_name: str) -> np.ndarray:
    try:
        return constants[name]
    except KeyError as error:
        raise CompileError(f"{op_name}: weight and bias operands must be captured constants") from error


def _unique_value_name(base: str, reserved: set[str]) -> str:
    if base not in reserved:
        reserved.add(base)
        return base
    suffix = 1
    while f"{base}.{suffix}" in reserved:
        suffix += 1
    result = f"{base}.{suffix}"
    reserved.add(result)
    return result


def _batch_norm_constant(
    node: Any,
    index: int,
    description: str,
    values: dict[str, FloatValue],
    constants: dict[str, np.ndarray],
    *,
    optional: bool,
) -> np.ndarray | None:
    operand = _arg(node, index, None)
    if operand is None:
        if optional:
            return None
        raise CompileError(f"{node.name}: BatchNorm {description} must be a captured constant")
    name = _tensor_name(operand, values, f"{node.name} {description}")
    try:
        return constants[name]
    except KeyError as error:
        raise CompileError(
            f"{node.name}: BatchNorm {description} must be a captured constant"
        ) from error


def _fold_batch_norm(
    node: Any,
    values: dict[str, FloatValue],
    constants: dict[str, np.ndarray],
    ops: list[object],
    reserved_names: set[str],
) -> None:
    """Fold one eval BatchNorm into its direct FP32 arithmetic producer."""

    training = _literal_bool(_arg(node, 5, False), f"{node.name} training")
    if training:
        raise CompileError(f"{node.name}: BatchNorm requires training=False for host folding")
    eps_arg = _arg(node, 7, 1e-5)
    if isinstance(eps_arg, bool) or not isinstance(eps_arg, Real):
        raise CompileError(f"{node.name}: BatchNorm eps must be a finite positive real")
    eps = float(eps_arg)
    if not math.isfinite(eps) or eps <= 0.0:
        raise CompileError(f"{node.name}: BatchNorm eps must be a finite positive real")

    producer_node = _arg(node, 0, None)
    producer_output = _tensor_name(producer_node, values, f"{node.name} input")
    users = getattr(producer_node, "users", None)
    if users is None or len(users) != 1:
        raise CompileError(
            f"{node.name}: BatchNorm producer must have exactly one consumer; shared/fan-out is unsupported"
        )
    if not ops or ops[-1].outputs != (producer_output,):  # type: ignore[attr-defined]
        raise CompileError(f"{node.name}: BatchNorm must directly follow its arithmetic producer")
    producer = ops[-1]
    if not isinstance(
        producer,
        (FloatConv1DOp, FloatConv2DOp, FloatDepthwiseConv2DOp, FloatLinearOp),
    ):
        raise CompileError(
            f"{node.name}: BatchNorm can only fold into Conv2D, DepthwiseConv2D, or Linear"
        )

    input_value = values[producer_output]
    output_value = values[node.name]
    expected_rank = (
        2
        if isinstance(producer, FloatLinearOp)
        else 3
        if isinstance(producer, FloatConv1DOp)
        else 4
    )
    if (
        len(input_value.shape) != expected_rank
        or output_value.shape != input_value.shape
        or output_value.layout is not input_value.layout
    ):
        raise CompileError(f"{node.name}: BatchNorm rank/shape/layout does not match its producer")
    channels = input_value.shape[1]

    gamma = _batch_norm_constant(
        node, 1, "weight", values, constants, optional=True
    )
    beta = _batch_norm_constant(
        node, 2, "bias", values, constants, optional=True
    )
    mean = _batch_norm_constant(
        node, 3, "running_mean", values, constants, optional=False
    )
    variance = _batch_norm_constant(
        node, 4, "running_var", values, constants, optional=False
    )
    assert mean is not None and variance is not None
    for description, array in (
        ("weight", gamma),
        ("bias", beta),
        ("running_mean", mean),
        ("running_var", variance),
    ):
        if array is not None and array.shape != (channels,):
            raise CompileError(
                f"{node.name}: BatchNorm {description} must contain {channels} channel values"
            )

    variance_plus_eps = variance.astype(np.float64) + eps
    if not np.all(np.isfinite(variance_plus_eps)) or np.any(variance_plus_eps <= 0.0):
        raise CompileError(f"{node.name}: BatchNorm running_var + eps must be finite and positive")
    gamma64 = (
        np.ones(channels, dtype=np.float64)
        if gamma is None
        else gamma.astype(np.float64)
    )
    beta64 = (
        np.zeros(channels, dtype=np.float64)
        if beta is None
        else beta.astype(np.float64)
    )
    scale = gamma64 / np.sqrt(variance_plus_eps)

    original_weight = _require_constant(constants, producer.weight, node.name)
    if original_weight.shape[0] != channels:
        raise CompileError(
            f"{node.name}: producer weight output channels do not match BatchNorm channels"
        )
    if producer.bias is None:
        original_bias = np.zeros(channels, dtype=np.float64)
    else:
        original_bias = _require_constant(constants, producer.bias, node.name).astype(np.float64)
        if original_bias.shape != (channels,):
            raise CompileError(f"{node.name}: producer bias does not match BatchNorm channels")
    scale_shape = (channels,) + (1,) * (original_weight.ndim - 1)
    folded_weight = (
        original_weight.astype(np.float64) * scale.reshape(scale_shape)
    ).astype(np.float32)
    folded_bias = ((original_bias - mean.astype(np.float64)) * scale + beta64).astype(
        np.float32
    )
    if not np.all(np.isfinite(folded_weight)) or not np.all(np.isfinite(folded_bias)):
        raise CompileError(f"{node.name}: BatchNorm folding produced NaN or infinity")

    weight_name = _unique_value_name(
        f"{producer.weight}.folded_bn.{node.name}.weight", reserved_names
    )
    bias_name = _unique_value_name(
        f"{producer.weight}.folded_bn.{node.name}.bias", reserved_names
    )
    weight_value = values[producer.weight]
    values[weight_name] = FloatValue(
        weight_name,
        tuple(folded_weight.shape),
        FloatDType.FLOAT32,
        weight_value.layout,
        FloatValueKind.CONSTANT,
    )
    values[bias_name] = FloatValue(
        bias_name,
        tuple(folded_bias.shape),
        FloatDType.FLOAT32,
        FloatLayout.C,
        FloatValueKind.CONSTANT,
    )
    constants[weight_name] = np.array(folded_weight, copy=True, order="C")
    constants[bias_name] = np.array(folded_bias, copy=True, order="C")
    del values[producer_output]
    ops[-1] = replace(
        producer,
        weight=weight_name,
        bias=bias_name,
        output=node.name,
    )


def _conv_output_shape(
    input_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    stride: tuple[int, int],
    padding: tuple[int, int, int, int],
    dilation: tuple[int, int],
) -> tuple[int, int, int, int]:
    _, _, input_h, input_w = input_shape
    output_channels, _, kernel_h, kernel_w = weight_shape
    effective_h = dilation[0] * (kernel_h - 1) + 1
    effective_w = dilation[1] * (kernel_w - 1) + 1
    output_h = (input_h + padding[0] + padding[1] - effective_h) // stride[0] + 1
    output_w = (input_w + padding[2] + padding[3] - effective_w) // stride[1] + 1
    return 1, output_channels, output_h, output_w


def _extract_conv(node: Any, values: dict[str, FloatValue], constants: dict[str, np.ndarray]) -> object:
    input_name = _tensor_name(_arg(node, 0, None), values, f"{node.name} input")
    weight_name = _tensor_name(_arg(node, 1, None), values, f"{node.name} weight")
    bias_arg = _arg(node, 2, None)
    bias_name = None if bias_arg is None else _tensor_name(bias_arg, values, f"{node.name} bias")
    stride = _pair(_arg(node, 3, (1, 1)), f"{node.name} stride", minimum=1)
    padding = _explicit_padding(_arg(node, 4, (0, 0)), f"{node.name} padding")
    dilation = _pair(_arg(node, 5, (1, 1)), f"{node.name} dilation", minimum=1)
    groups = _literal_int(_arg(node, 6, 1), f"{node.name} groups")

    input_value = values[input_name]
    output_value = values[node.name]
    weight = _require_constant(constants, weight_name, node.name)
    if input_value.layout is not FloatLayout.NCHW or weight.ndim != 4:
        raise CompileError(f"{node.name}: Conv2d requires rank-four NCHW input and OIHW weight")
    input_channels = input_value.shape[1]
    output_channels, grouped_channels, _, _ = weight.shape
    if bias_name is not None:
        bias = _require_constant(constants, bias_name, node.name)
        if bias.shape != (output_channels,):
            raise CompileError(f"{node.name}: Conv2d bias shape must equal output channels")
        _set_layout(values, bias_name, FloatLayout.C)
    _set_layout(values, weight_name, FloatLayout.OIHW)
    if output_value.shape != _conv_output_shape(
        input_value.shape, tuple(weight.shape), stride, padding, dilation
    ):
        raise CompileError(f"{node.name}: exported Conv2d output shape violates static formula")

    if groups <= 0 or input_channels % groups or output_channels % groups:
        raise CompileError(f"{node.name}: Conv2d groups must divide input and output channels")
    if groups == 1:
        if grouped_channels != input_channels:
            raise CompileError(f"{node.name}: Conv2d weight input channels do not match input")
        return FloatConv2DOp(
            node.name,
            input_name,
            weight_name,
            bias_name,
            node.name,
            stride,
            padding,
            dilation,
            groups,
        )
    if groups == input_channels and grouped_channels == 1 and output_channels % input_channels == 0:
        return FloatDepthwiseConv2DOp(
            node.name,
            input_name,
            weight_name,
            bias_name,
            node.name,
            stride,
            padding,
            dilation,
            input_channels,
            output_channels // input_channels,
        )
    if grouped_channels != input_channels // groups:
        raise CompileError(
            f"{node.name}: grouped Conv2d weight channels must equal input_channels/groups"
        )
    return FloatConv2DOp(
        node.name,
        input_name,
        weight_name,
        bias_name,
        node.name,
        stride,
        padding,
        dilation,
        groups,
    )


def _extract_conv1d(
    node: Any,
    values: dict[str, FloatValue],
    constants: dict[str, np.ndarray],
) -> FloatConv1DOp:
    input_name = _tensor_name(_arg(node, 0, None), values, f"{node.name} input")
    weight_name = _tensor_name(_arg(node, 1, None), values, f"{node.name} weight")
    bias_arg = _arg(node, 2, None)
    bias_name = None if bias_arg is None else _tensor_name(bias_arg, values, f"{node.name} bias")
    stride = _literal_1d(_arg(node, 3, (1,)), f"{node.name} stride", minimum=1)
    pad = _literal_1d(_arg(node, 4, (0,)), f"{node.name} padding", minimum=0)
    dilation = _literal_1d(_arg(node, 5, (1,)), f"{node.name} dilation", minimum=1)
    groups = _literal_int(_arg(node, 6, 1), f"{node.name} groups")
    input_value, output_value = values[input_name], values[node.name]
    weight = _require_constant(constants, weight_name, node.name)
    if input_value.layout is not FloatLayout.NCL or weight.ndim != 3:
        raise CompileError(f"{node.name}: Conv1d requires rank-three NCL input and OIW weight")
    input_channels = input_value.shape[1]
    output_channels, grouped_channels, kernel = weight.shape
    if groups <= 0 or input_channels % groups or output_channels % groups:
        raise CompileError(f"{node.name}: Conv1d groups must divide input and output channels")
    if grouped_channels != input_channels // groups:
        raise CompileError(f"{node.name}: Conv1d weight channels must equal input_channels/groups")
    effective = dilation * (kernel - 1) + 1
    output_length = (input_value.shape[2] + 2 * pad - effective) // stride + 1
    if output_value.shape != (1, output_channels, output_length):
        raise CompileError(f"{node.name}: exported Conv1d output shape violates static formula")
    _set_layout(values, weight_name, FloatLayout.OIW)
    if bias_name is not None:
        bias = _require_constant(constants, bias_name, node.name)
        if bias.shape != (output_channels,):
            raise CompileError(f"{node.name}: Conv1d bias shape must equal output channels")
        _set_layout(values, bias_name, FloatLayout.C)
    return FloatConv1DOp(
        node.name,
        input_name,
        weight_name,
        bias_name,
        node.name,
        stride,
        (pad, pad),
        dilation,
        groups,
    )


def _extract_conv_transpose2d(
    node: Any,
    values: dict[str, FloatValue],
    constants: dict[str, np.ndarray],
) -> FloatConvTranspose2DOp:
    input_name = _tensor_name(_arg(node, 0, None), values, f"{node.name} input")
    weight_name = _tensor_name(_arg(node, 1, None), values, f"{node.name} weight")
    bias_arg = _arg(node, 2, None)
    bias_name = None if bias_arg is None else _tensor_name(bias_arg, values, f"{node.name} bias")
    stride = _pair(_arg(node, 3, (1, 1)), f"{node.name} stride", minimum=1)
    pad = _pair(_arg(node, 4, (0, 0)), f"{node.name} padding", minimum=0)
    output_padding = _pair(_arg(node, 5, (0, 0)), f"{node.name} output_padding", minimum=0)
    groups = _literal_int(_arg(node, 6, 1), f"{node.name} groups")
    dilation = _pair(_arg(node, 7, (1, 1)), f"{node.name} dilation", minimum=1)
    if groups <= 0:
        raise CompileError(f"{node.name}: ConvTranspose2D groups must be positive")
    input_value, output_value = values[input_name], values[node.name]
    weight = _require_constant(constants, weight_name, node.name)
    if input_value.layout is not FloatLayout.NCHW or output_value.layout is not FloatLayout.NCHW:
        raise CompileError(f"{node.name}: ConvTranspose2D requires NCHW activations")
    if (
        weight.ndim != 4
        or input_value.shape[1] % groups != 0
        or output_value.shape[1] % groups != 0
        or weight.shape[0] != input_value.shape[1]
        or weight.shape[1] * groups != output_value.shape[1]
    ):
        raise CompileError(f"{node.name}: ConvTranspose2D IOHW weight channels are incompatible")
    _set_layout(values, weight_name, FloatLayout.IOHW)
    if bias_name is not None:
        bias = _require_constant(constants, bias_name, node.name)
        if bias.shape != (output_value.shape[1],):
            raise CompileError(f"{node.name}: ConvTranspose2D bias shape must equal output channels")
        _set_layout(values, bias_name, FloatLayout.C)
    padding = (pad[0], pad[0], pad[1], pad[1])
    expected = (
        1,
        output_value.shape[1],
        (input_value.shape[2] - 1) * stride[0] - 2 * pad[0]
        + dilation[0] * (weight.shape[2] - 1) + output_padding[0] + 1,
        (input_value.shape[3] - 1) * stride[1] - 2 * pad[1]
        + dilation[1] * (weight.shape[3] - 1) + output_padding[1] + 1,
    )
    if output_value.shape != expected:
        raise CompileError(f"{node.name}: ConvTranspose2D output shape is invalid")
    return FloatConvTranspose2DOp(
        node.name, input_name, weight_name, bias_name, node.name,
        stride, padding, output_padding, dilation, groups,
    )


def _extract_linear(node: Any, values: dict[str, FloatValue], constants: dict[str, np.ndarray]) -> FloatLinearOp:
    input_name = _tensor_name(_arg(node, 0, None), values, f"{node.name} input")
    weight_name = _tensor_name(_arg(node, 1, None), values, f"{node.name} weight")
    bias_arg = _arg(node, 2, None)
    bias_name = None if bias_arg is None else _tensor_name(bias_arg, values, f"{node.name} bias")
    input_value = values[input_name]
    output_value = values[node.name]
    weight = _require_constant(constants, weight_name, node.name)
    if input_value.layout is not FloatLayout.NC or weight.ndim != 2:
        raise CompileError(f"{node.name}: Linear requires rank-two NC input and OI weight")
    if tuple(weight.shape) != (output_value.shape[1], input_value.shape[1]):
        raise CompileError(f"{node.name}: Linear shapes are incompatible")
    _set_layout(values, weight_name, FloatLayout.OI)
    if bias_name is not None:
        bias = _require_constant(constants, bias_name, node.name)
        if bias.shape != (output_value.shape[1],):
            raise CompileError(f"{node.name}: Linear bias shape must equal output channels")
        _set_layout(values, bias_name, FloatLayout.C)
    return FloatLinearOp(node.name, input_name, weight_name, bias_name, node.name)


def _extract_binary(node: Any, values: dict[str, FloatValue], *, multiply: bool) -> FloatAddOp | FloatMulOp:
    input_a = _tensor_name(_arg(node, 0, None), values, f"{node.name} input_a")
    input_b = _tensor_name(_arg(node, 1, None), values, f"{node.name} input_b")
    shape_a = values[input_a].shape
    shape_b = values[input_b].shape
    if len(shape_a) != len(shape_b):
        raise CompileError(f"{node.name}: static Add/Mul broadcasting requires equal ranks")
    if any(a != b and a != 1 and b != 1 for a, b in zip(shape_a, shape_b)):
        raise CompileError(f"{node.name}: Add/Mul input shapes are not broadcast-compatible")
    expected = tuple(max(a, b) for a, b in zip(shape_a, shape_b))
    if values[node.name].shape != expected:
        raise CompileError(f"{node.name}: Add/Mul output does not match broadcast shape {expected}")
    if values[input_a].layout is not values[input_b].layout or values[node.name].layout is not values[input_a].layout:
        raise CompileError(f"{node.name}: Add/Mul layouts must match exactly")
    if not multiply:
        alpha = node.kwargs.get("alpha", _arg(node, 2, 1))
        if isinstance(alpha, bool) or not isinstance(alpha, Real) or float(alpha) != 1.0:
            raise CompileError(f"{node.name}: Add alpha must equal 1")
        return FloatAddOp(node.name, input_a, input_b, node.name)
    return FloatMulOp(node.name, input_a, input_b, node.name)


def _extract_inplace_add(
    node: Any, values: dict[str, FloatValue]
) -> FloatAddOp:
    target_input = _arg(node, 0, None)
    target_name = _tensor_name(target_input, values, f"{node.name} target input")
    if values[target_name].kind in (
        FloatValueKind.INPUT,
        FloatValueKind.PARAMETER,
        FloatValueKind.BUFFER,
        FloatValueKind.CONSTANT,
    ):
        raise CompileError(
            f"{node.name}: aten.add_.Tensor may not mutate caller or captured constant storage"
        )
    users = getattr(target_input, "users", None)
    if users is None or len(users) != 1 or node not in users:
        raise CompileError(
            f"{node.name}: aten.add_.Tensor target input is shared/fan-out; "
            "mutation semantics are unsupported"
        )
    # FX proves the mutated value has no other semantic consumer, and the
    # exported signature was already checked for caller/buffer mutation. It is
    # therefore safe to normalize this target to a new immutable SSA result.
    result = _extract_binary(node, values, multiply=False)
    assert isinstance(result, FloatAddOp)
    return result


def _extract_pool(node: Any, values: dict[str, FloatValue], *, average: bool) -> object:
    input_name = _tensor_name(_arg(node, 0, None), values, f"{node.name} input")
    kernel = _pair(_arg(node, 1, None), f"{node.name} kernel", minimum=1)
    stride_arg = _arg(node, 2, kernel)
    if isinstance(stride_arg, (tuple, list)) and len(stride_arg) == 0:
        stride_arg = kernel
    stride = _pair(stride_arg, f"{node.name} stride", minimum=1)
    padding = _explicit_padding(_arg(node, 3, (0, 0)), f"{node.name} padding")
    if average:
        ceil_mode = _literal_bool(_arg(node, 4, False), f"{node.name} ceil_mode")
        count_include_pad = _literal_bool(
            _arg(node, 5, True), f"{node.name} count_include_pad"
        )
        divisor_override = _arg(node, 6, None)
        if divisor_override is not None:
            raise CompileError(f"{node.name}: AvgPool divisor_override is unsupported")
        if count_include_pad and any(padding):
            raise CompileError(
                f"{node.name}: P0 AveragePool excludes padded coordinates; set count_include_pad=False"
            )
    else:
        dilation = _pair(_arg(node, 4, (1, 1)), f"{node.name} dilation", minimum=1)
        if dilation != (1, 1):
            raise CompileError(f"{node.name}: MaxPool dilation is unsupported")
        ceil_mode = _literal_bool(_arg(node, 5, False), f"{node.name} ceil_mode")
    if ceil_mode:
        raise CompileError(f"{node.name}: ceil_mode pooling is unsupported")
    input_value = values[input_name]
    output_value = values[node.name]
    if input_value.layout is not FloatLayout.NCHW or output_value.layout is not FloatLayout.NCHW:
        raise CompileError(f"{node.name}: Pool2d requires rank-four NCHW tensors")
    expected_h = (input_value.shape[2] + padding[0] + padding[1] - kernel[0]) // stride[0] + 1
    expected_w = (input_value.shape[3] + padding[2] + padding[3] - kernel[1]) // stride[1] + 1
    if output_value.shape != (1, input_value.shape[1], expected_h, expected_w):
        raise CompileError(f"{node.name}: exported Pool2d output shape violates static formula")
    op_type = FloatAveragePool2DOp if average else FloatMaxPool2DOp
    return op_type(node.name, input_name, node.name, kernel, stride, padding)


def _extract_pool1d(node: Any, values: dict[str, FloatValue], *, average: bool) -> object:
    input_name = _tensor_name(_arg(node, 0, None), values, f"{node.name} input")
    kernel = _literal_1d(_arg(node, 1, None), f"{node.name} kernel", minimum=1)
    stride_arg = _arg(node, 2, (kernel,))
    if isinstance(stride_arg, (tuple, list)) and len(stride_arg) == 0:
        stride_arg = (kernel,)
    stride = _literal_1d(stride_arg, f"{node.name} stride", minimum=1)
    pad = _literal_1d(_arg(node, 3, (0,)), f"{node.name} padding", minimum=0)
    if average:
        ceil_mode = _literal_bool(_arg(node, 4, False), f"{node.name} ceil_mode")
        count_include_pad = _literal_bool(_arg(node, 5, True), f"{node.name} count_include_pad")
        if count_include_pad and pad:
            raise CompileError(
                f"{node.name}: AveragePool1D excludes padded positions; set count_include_pad=False"
            )
    else:
        ceil_mode = False
    if ceil_mode:
        raise CompileError(f"{node.name}: Pool1D ceil_mode is unsupported")
    input_value, output_value = values[input_name], values[node.name]
    if input_value.layout is not FloatLayout.NCL or output_value.layout is not FloatLayout.NCL:
        raise CompileError(f"{node.name}: Pool1D requires rank-three NCL tensors")
    output_length = (input_value.shape[2] + 2 * pad - kernel) // stride + 1
    if output_value.shape != (1, input_value.shape[1], output_length):
        raise CompileError(f"{node.name}: exported Pool1D output shape violates static formula")
    op_type = FloatAveragePool1DOp if average else FloatMaxPool1DOp
    return op_type(node.name, input_name, node.name, kernel, stride, (pad, pad))


def _extract_adaptive_global_average_pool(
    node: Any, values: dict[str, FloatValue]
) -> FloatAveragePool2DOp:
    input_name = _tensor_name(_arg(node, 0, None), values, f"{node.name} input")
    output_size = _pair(_arg(node, 1, None), f"{node.name} output_size", minimum=1)
    if output_size != (1, 1):
        raise CompileError(
            f"{node.name}: P0 AdaptiveAvgPool2d supports global output_size=(1, 1) only"
        )
    input_value = values[input_name]
    output_value = values[node.name]
    if input_value.layout is not FloatLayout.NCHW or len(input_value.shape) != 4:
        raise CompileError(f"{node.name}: AdaptiveAvgPool2d requires rank-four NCHW input")
    if output_value.shape != (1, input_value.shape[1], 1, 1):
        raise CompileError(f"{node.name}: AdaptiveAvgPool2d output shape is invalid")
    kernel = (input_value.shape[2], input_value.shape[3])
    return FloatAveragePool2DOp(
        node.name,
        input_name,
        node.name,
        kernel,
        kernel,
        (0, 0, 0, 0),
    )


def _require_unshared_inplace_input(node: Any, description: str) -> None:
    input_node = _arg(node, 0, None)
    users = getattr(input_node, "users", None)
    if users is None or len(users) != 1 or node not in users:
        raise CompileError(
            f"{node.name}: {description} input is shared/fan-out; mutation semantics are unsupported"
        )


def _extract_relu_like(node: Any, values: dict[str, FloatValue], target: str) -> object:
    input_name = _tensor_name(_arg(node, 0, None), values, f"{node.name} input")
    if values[input_name].shape != values[node.name].shape:
        raise CompileError(f"{node.name}: activation must preserve shape")
    if target in ("aten.relu_.default", "aten.hardtanh_.default"):
        _require_unshared_inplace_input(node, target)
    if target in ("aten.hardtanh.default", "aten.hardtanh_.default"):
        minimum = _arg(node, 1, -1.0)
        maximum = _arg(node, 2, 1.0)
        if (
            isinstance(minimum, bool)
            or not isinstance(minimum, Real)
            or isinstance(maximum, bool)
            or not isinstance(maximum, Real)
            or not math.isfinite(float(minimum))
            or not math.isfinite(float(maximum))
            or float(minimum) != 0.0
            or float(maximum) != 6.0
        ):
            raise CompileError(
                f"{node.name}: Hardtanh is supported only for static min=0 and max=6 (ReLU6)"
            )
        return FloatReLU6Op(node.name, input_name, node.name)
    if target == "aten.relu6.default":
        return FloatReLU6Op(node.name, input_name, node.name)
    return FloatReLUOp(node.name, input_name, node.name)


def _extract_nonlinear(node: Any, values: dict[str, FloatValue], target: str) -> object:
    input_name = _tensor_name(_arg(node, 0, None), values, f"{node.name} input")
    if values[input_name].shape != values[node.name].shape:
        raise CompileError(f"{node.name}: nonlinear activation must preserve shape")
    if target.endswith("_.default"):
        _require_unshared_inplace_input(node, target)
    op_type = {
        "aten.sigmoid.default": FloatSigmoidOp,
        "aten.hardswish.default": FloatHardSwishOp,
        "aten.hardswish_.default": FloatHardSwishOp,
        "aten.hardsigmoid.default": FloatHardSigmoidOp,
        "aten.hardsigmoid_.default": FloatHardSigmoidOp,
        "aten.silu.default": FloatSiLUOp,
        "aten.silu_.default": FloatSiLUOp,
    }[target]
    return op_type(node.name, input_name, node.name)


def _extract_pad2d(node: Any, values: dict[str, FloatValue]) -> FloatPad2DOp:
    input_name = _tensor_name(_arg(node, 0, None), values, f"{node.name} input")
    raw_padding = _arg(node, 1, None)
    try:
        padding = tuple(raw_padding)  # type: ignore[arg-type]
    except TypeError as error:
        raise CompileError(f"{node.name}: Pad padding must be static") from error
    if len(padding) != 4 or any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in padding):
        raise CompileError(f"{node.name}: Pad2D requires four non-negative spatial values")
    mode = _arg(node, 2, "constant")
    value = _arg(node, 3, 0.0)
    if mode != "constant" or isinstance(value, bool) or not isinstance(value, Real) or float(value) != 0.0:
        raise CompileError(f"{node.name}: Pad2D supports constant real-zero padding only")
    input_value, output_value = values[input_name], values[node.name]
    if input_value.layout is not FloatLayout.NCHW or output_value.layout is not FloatLayout.NCHW:
        raise CompileError(f"{node.name}: Pad2D requires NCHW tensors")
    left, right, top, bottom = (int(x) for x in padding)
    expected = (1, input_value.shape[1], input_value.shape[2] + top + bottom, input_value.shape[3] + left + right)
    if output_value.shape != expected:
        raise CompileError(f"{node.name}: Pad2D output shape is invalid")
    return FloatPad2DOp(node.name, input_name, node.name, (top, bottom, left, right))


def _extract_reduce_mean(node: Any, values: dict[str, FloatValue]) -> FloatReduceMeanOp:
    input_name = _tensor_name(_arg(node, 0, None), values, f"{node.name} input")
    raw_axes = _arg(node, 1, None)
    try:
        axes = tuple(_literal_int(axis, f"{node.name} axis") for axis in raw_axes)  # type: ignore[union-attr]
    except TypeError as error:
        raise CompileError(f"{node.name}: ReduceMean axes must be static") from error
    keepdims = _literal_bool(_arg(node, 2, False), f"{node.name} keepdim")
    input_value, output_value = values[input_name], values[node.name]
    rank = len(input_value.shape)
    normalized = tuple(sorted(axis % rank for axis in axes))
    expected_axes = (2, 3) if input_value.layout is FloatLayout.NCHW else (2,) if input_value.layout is FloatLayout.NCL else ()
    if normalized != expected_axes:
        raise CompileError(f"{node.name}: ReduceMean supports NCHW spatial or NCL time axes")
    expected = tuple(
        1 if index in normalized else size
        for index, size in enumerate(input_value.shape)
        if keepdims or index not in normalized
    )
    if output_value.shape != expected:
        raise CompileError(f"{node.name}: ReduceMean output shape is invalid")
    if not keepdims and output_value.layout is not FloatLayout.NC:
        raise CompileError(f"{node.name}: reduced spatial/time output must use NC layout")
    return FloatReduceMeanOp(node.name, input_name, node.name, axes, keepdims)


def _extract_resize2d(
    node: Any, values: dict[str, FloatValue], *, bilinear: bool
) -> FloatResizeNearest2DOp | FloatResizeBilinear2DOp:
    input_name = _tensor_name(_arg(node, 0, None), values, f"{node.name} input")
    source = values[input_name]
    output = values[node.name]
    if source.layout is not FloatLayout.NCHW or output.layout is not FloatLayout.NCHW:
        raise CompileError(f"{node.name}: Resize2D requires NCHW tensors")
    if source.shape[0] != 1 or output.shape[0] != 1 or source.shape[1] != output.shape[1]:
        raise CompileError(f"{node.name}: Resize2D preserves batch and channels")
    raw_size = _arg(node, 1, None)
    raw_scales = _arg(node, 3 if bilinear else 2, None)
    if raw_size is not None:
        size = _pair(raw_size, f"{node.name} output_size", minimum=1)
        if size != output.shape[2:4]:
            raise CompileError(f"{node.name}: Resize2D output_size does not match metadata")
    elif raw_scales is not None:
        try:
            scales = tuple(raw_scales)
        except TypeError as error:
            raise CompileError(f"{node.name}: Resize2D scale_factors must be static") from error
        if len(scales) != 2 or any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            or not float(value).is_integer()
            for value in scales
        ):
            raise CompileError(
                f"{node.name}: Resize2D currently requires positive integer scale_factors"
            )
        expected = (
            source.shape[2] * int(scales[0]),
            source.shape[3] * int(scales[1]),
        )
        if output.shape[2:4] != expected:
            raise CompileError(f"{node.name}: Resize2D output does not match scale_factors")
    else:
        raise CompileError(f"{node.name}: Resize2D requires static output_size or scale_factors")
    if bilinear:
        align_corners = _literal_bool(_arg(node, 2, False), f"{node.name} align_corners")
        return FloatResizeBilinear2DOp(node.name, input_name, node.name, align_corners)
    return FloatResizeNearest2DOp(node.name, input_name, node.name)


def _extract_squeeze_view(node: Any, values: dict[str, FloatValue]) -> FloatReshapeOp:
    input_name = _tensor_name(_arg(node, 0, None), values, f"{node.name} input")
    dimension = _literal_int(_arg(node, 1, None), f"{node.name} dimension")
    input_value, output_value = values[input_name], values[node.name]
    target = str(node.target)
    if target == "aten.squeeze.dim":
        if not -len(input_value.shape) <= dimension < len(input_value.shape):
            raise CompileError(f"{node.name}: Squeeze dimension is outside input rank")
        normalized = dimension % len(input_value.shape)
        if input_value.shape[normalized] != 1:
            raise CompileError(f"{node.name}: Squeeze dimension must be statically one")
        expected = input_value.shape[:normalized] + input_value.shape[normalized + 1 :]
    else:
        output_rank = len(input_value.shape) + 1
        normalized = dimension + output_rank if dimension < 0 else dimension
        if not 0 <= normalized <= len(input_value.shape):
            raise CompileError(f"{node.name}: Unsqueeze dimension is outside output rank")
        expected = input_value.shape[:normalized] + (1,) + input_value.shape[normalized:]
    if output_value.shape != expected or math.prod(input_value.shape) != math.prod(output_value.shape):
        raise CompileError(f"{node.name}: Squeeze/Unsqueeze output shape is invalid")
    return FloatReshapeOp(node.name, input_name, node.name, output_value.shape)


def _extract_slice(node: Any, values: dict[str, FloatValue]) -> FloatSliceOp:
    input_name = _tensor_name(_arg(node, 0, None), values, f"{node.name} input")
    input_value = values[input_name]
    output_value = values[node.name]
    rank = len(input_value.shape)
    raw_axis = _literal_int(_arg(node, 1, 0), f"{node.name} axis")
    if not -rank <= raw_axis < rank:
        raise CompileError(f"{node.name}: Slice axis is outside input rank")
    axis = raw_axis % rank

    def optional_bound(value: object, description: str) -> int | None:
        if value is None:
            return None
        return _literal_int(value, description)

    start = optional_bound(_arg(node, 2, None), f"{node.name} start")
    stop = optional_bound(_arg(node, 3, None), f"{node.name} stop")
    step = _literal_int(_arg(node, 4, 1), f"{node.name} step")
    if step <= 0:
        raise CompileError(f"{node.name}: Slice requires a positive static step")
    normalized_start, normalized_stop, normalized_step = slice(
        start, stop, step
    ).indices(input_value.shape[axis])
    if normalized_step > (1 << 31) - 1:
        raise CompileError(f"{node.name}: Slice step exceeds the portable target limit")
    output_axis_size = (
        normalized_stop - normalized_start + normalized_step - 1
    ) // normalized_step
    if output_axis_size <= 0:
        raise CompileError(f"{node.name}: empty Slice outputs are unsupported")
    expected = list(input_value.shape)
    expected[axis] = output_axis_size
    if output_value.shape != tuple(expected) or output_value.layout is not input_value.layout:
        raise CompileError(f"{node.name}: Slice output shape/layout is invalid")
    return FloatSliceOp(
        node.name,
        input_name,
        node.name,
        axis,
        normalized_start,
        normalized_stop,
        normalized_step,
    )


def _remove_eval_dropout(node: Any, values: dict[str, FloatValue]) -> None:
    input_name = _tensor_name(_arg(node, 0, None), values, f"{node.name} input")
    probability = _arg(node, 1, 0.5)
    if (
        isinstance(probability, bool)
        or not isinstance(probability, Real)
        or not math.isfinite(float(probability))
        or not 0.0 <= float(probability) <= 1.0
    ):
        raise CompileError(f"{node.name}: Dropout probability must be a finite value in [0, 1]")
    training = _literal_bool(_arg(node, 2, True), f"{node.name} training")
    if training:
        raise CompileError(f"{node.name}: Dropout requires training=False for inference removal")
    if values[node.name].shape != values[input_name].shape:
        raise CompileError(f"{node.name}: eval Dropout must preserve input shape")
    # Keep a temporary alias entry so later FX nodes resolve this edge directly
    # to input_name. Alias keys are filtered before FloatGraph construction.
    values[node.name] = values[input_name]


def _extract_reshape(node: Any, values: dict[str, FloatValue], *, flatten: bool) -> object:
    input_name = _tensor_name(_arg(node, 0, None), values, f"{node.name} input")
    input_value = values[input_name]
    output_value = values[node.name]
    if math.prod(input_value.shape) != math.prod(output_value.shape):
        raise CompileError(f"{node.name}: reshape changes the number of elements")
    if flatten:
        start_dim = _literal_int(_arg(node, 1, 0), f"{node.name} start_dim")
        end_dim = _literal_int(_arg(node, 2, -1), f"{node.name} end_dim")
        normalized_end = end_dim + len(input_value.shape) if end_dim < 0 else end_dim
        if start_dim != 1 or normalized_end != len(input_value.shape) - 1:
            raise CompileError(f"{node.name}: P0 Flatten must preserve batch and flatten dimensions 1..last")
        if output_value.shape != (1, math.prod(input_value.shape[1:])):
            raise CompileError(f"{node.name}: Flatten output shape is invalid")
        return FloatFlattenOp(node.name, input_name, node.name, 1, -1)
    requested = _arg(node, 1, None)
    try:
        requested_items = tuple(requested)  # type: ignore[arg-type]
    except TypeError as error:
        raise CompileError(f"{node.name}: Reshape target must be a static integer sequence") from error
    if not requested_items or any(
        isinstance(item, bool) or not isinstance(item, Integral) for item in requested_items
    ):
        raise CompileError(f"{node.name}: Reshape target must be a static integer sequence")
    return FloatReshapeOp(node.name, input_name, node.name, output_value.shape)


def _extract_concat(node: Any, values: dict[str, FloatValue]) -> FloatConcatOp:
    sequence = _arg(node, 0, None)
    if not isinstance(sequence, (tuple, list)) or len(sequence) < 2:
        raise CompileError(f"{node.name}: Concat requires at least two tensor inputs")
    inputs = tuple(_tensor_name(value, values, f"{node.name} input") for value in sequence)
    rank = len(values[inputs[0]].shape)
    axis = _literal_int(_arg(node, 1, 0), f"{node.name} axis")
    axis = axis + rank if axis < 0 else axis
    if not 0 <= axis < rank:
        raise CompileError(f"{node.name}: Concat axis is outside tensor rank")
    reference = values[inputs[0]]
    for input_name in inputs[1:]:
        value = values[input_name]
        if len(value.shape) != rank or value.layout is not reference.layout:
            raise CompileError(f"{node.name}: Concat ranks/layouts must match")
        if any(
            value.shape[index] != reference.shape[index]
            for index in range(rank)
            if index != axis
        ):
            raise CompileError(f"{node.name}: non-concatenated dimensions must match")
    expected = list(reference.shape)
    expected[axis] = sum(values[name].shape[axis] for name in inputs)
    if tuple(expected) != values[node.name].shape:
        raise CompileError(f"{node.name}: Concat output shape is invalid")
    return FloatConcatOp(node.name, inputs, node.name, axis)


def _extract_softmax(node: Any, values: dict[str, FloatValue]) -> FloatSoftmaxOp:
    input_name = _tensor_name(_arg(node, 0, None), values, f"{node.name} input")
    axis = _literal_int(_arg(node, 1, None), f"{node.name} axis")
    dtype = _arg(node, 2, None)
    input_value = values[input_name]
    normalized_axis = axis + len(input_value.shape) if axis < 0 else axis
    if input_value.layout is not FloatLayout.NC or normalized_axis != len(input_value.shape) - 1:
        raise CompileError(f"{node.name}: P0 Softmax requires rank-two NC input and final axis")
    if dtype is not None:
        raise CompileError(f"{node.name}: Softmax dtype conversion is unsupported")
    return FloatSoftmaxOp(node.name, input_name, node.name, -1)


def _extract_op(node: Any, values: dict[str, FloatValue], constants: dict[str, np.ndarray]) -> object:
    target = str(node.target)
    if target not in ALLOWED_ATEN_TARGETS:
        raise CompileError(
            f"{node.name}: unsupported torch.export operator {target}; no fallback is permitted"
        )
    if target == "aten.conv2d.default":
        return _extract_conv(node, values, constants)
    if target == "aten.conv1d.default":
        return _extract_conv1d(node, values, constants)
    if target == "aten.conv_transpose2d.input":
        return _extract_conv_transpose2d(node, values, constants)
    if target == "aten.batch_norm.default":
        raise AssertionError("BatchNorm must be folded before ordinary operation extraction")
    if target in ("aten.dropout.default", "aten.dropout_.default"):
        raise AssertionError("eval Dropout must be removed before ordinary operation extraction")
    if target == "aten.linear.default":
        return _extract_linear(node, values, constants)
    if target == "aten.add.Tensor":
        return _extract_binary(node, values, multiply=False)
    if target == "aten.add_.Tensor":
        return _extract_inplace_add(node, values)
    if target == "aten.mul.Tensor":
        return _extract_binary(node, values, multiply=True)
    if target in (
        "aten.relu.default",
        "aten.relu_.default",
        "aten.relu6.default",
        "aten.hardtanh.default",
        "aten.hardtanh_.default",
    ):
        return _extract_relu_like(node, values, target)
    if target in (
        "aten.sigmoid.default",
        "aten.hardswish.default",
        "aten.hardswish_.default",
        "aten.hardsigmoid.default",
        "aten.hardsigmoid_.default",
        "aten.silu.default",
        "aten.silu_.default",
    ):
        return _extract_nonlinear(node, values, target)
    if target == "aten.avg_pool1d.default":
        return _extract_pool1d(node, values, average=True)
    if target == "aten.max_pool1d.default":
        return _extract_pool1d(node, values, average=False)
    if target == "aten.avg_pool2d.default":
        return _extract_pool(node, values, average=True)
    if target == "aten.max_pool2d.default":
        return _extract_pool(node, values, average=False)
    if target == "aten.adaptive_avg_pool2d.default":
        return _extract_adaptive_global_average_pool(node, values)
    if target == "aten.flatten.using_ints":
        return _extract_reshape(node, values, flatten=True)
    if target in ("aten.reshape.default", "aten.view.default"):
        return _extract_reshape(node, values, flatten=False)
    if target in ("aten.squeeze.dim", "aten.unsqueeze.default"):
        return _extract_squeeze_view(node, values)
    if target == "aten.slice.Tensor":
        return _extract_slice(node, values)
    if target == "aten.pad.default":
        return _extract_pad2d(node, values)
    if target == "aten.mean.dim":
        return _extract_reduce_mean(node, values)
    if target == "aten.upsample_nearest2d.vec":
        return _extract_resize2d(node, values, bilinear=False)
    if target == "aten.upsample_bilinear2d.vec":
        return _extract_resize2d(node, values, bilinear=True)
    if target == "aten.cat.default":
        return _extract_concat(node, values)
    if target == "aten.softmax.int":
        return _extract_softmax(node, values)
    raise AssertionError(f"unhandled allowlisted operator {target}")


def capture_torch_export(
    model: object,
    example_input: object,
    *,
    name: str | None = None,
    dynamic_shapes: object | None = None,
) -> FloatGraph:
    """Capture one static FP32 eval graph through the real ``torch.export`` API.

    This stage performs typed semantic extraction and folds eval BatchNorm into
    direct single-consumer Conv/Depthwise/Linear constants. It intentionally
    does not calibrate, quantize, canonicalize NCHW to NHWC, or build
    QuantizedGraph.
    """

    torch = _load_torch()
    if dynamic_shapes is not None:
        raise CompileError("dynamic shapes are unsupported in P0; provide a fully static example")
    if not isinstance(model, torch.nn.Module):
        raise CompileError("torch.export frontend requires a torch.nn.Module")
    training_modules = [
        module_name or "<root>"
        for module_name, module in model.named_modules()
        if bool(module.training)
    ]
    if training_modules:
        raise CompileError(
            "torch.export frontend requires eval mode; training modules: "
            + ", ".join(training_modules)
        )
    args = example_input if isinstance(example_input, tuple) else (example_input,)
    if len(args) != 1 or not isinstance(args[0], torch.Tensor):
        raise CompileError("P0 requires exactly one tensor example input")
    input_tensor = args[0]
    if input_tensor.dtype is not torch.float32:
        raise CompileError(f"P0 example input must be float32, got {input_tensor.dtype}")
    if input_tensor.ndim not in (2, 3, 4) or input_tensor.shape[0] != 1:
        raise CompileError("example input must be static batch-one NC, NCL, or NCHW")
    if not bool(torch.isfinite(input_tensor).all().item()):
        raise CompileError("example input contains NaN or infinity")
    try:
        with torch.no_grad():
            exported = torch.export.export(model, args, strict=True)
    except Exception as error:
        raise CompileError(f"torch.export capture failed: {error}") from error
    mutation_outputs = [
        spec.kind.name
        for spec in exported.graph_signature.output_specs
        if spec.kind.name != "USER_OUTPUT"
    ]
    if mutation_outputs:
        raise CompileError(
            "torch.export input/buffer mutation outputs are unsupported: "
            + ", ".join(mutation_outputs)
        )

    spec_by_name = {
        spec.arg.name: spec
        for spec in exported.graph_signature.input_specs
        if hasattr(spec.arg, "name")
    }
    values: dict[str, FloatValue] = {}
    constants: dict[str, np.ndarray] = {}
    graph_inputs: list[str] = []
    ops: list[object] = []
    output_node: Any | None = None
    reserved_value_names = {
        node.name for node in exported.graph_module.graph.nodes if isinstance(node.name, str)
    }

    for node in exported.graph_module.graph.nodes:
        if node.op == "placeholder":
            spec = spec_by_name.get(node.name)
            if spec is None:
                raise CompileError(f"{node.name}: torch.export placeholder has no graph signature")
            kind = spec.kind.name
            if kind == "USER_INPUT":
                value = _activation_value(torch, node)
                values[node.name] = replace(value, kind=FloatValueKind.INPUT)
                graph_inputs.append(node.name)
                continue
            if kind not in ("PARAMETER", "BUFFER", "CONSTANT_TENSOR"):
                raise CompileError(f"{node.name}: unsupported torch.export input kind {kind}")
            target = spec.target
            if not isinstance(target, str) or not target:
                raise CompileError(f"{node.name}: lifted constant has no source target")
            if kind in ("PARAMETER", "BUFFER"):
                source = exported.state_dict.get(target)
            else:
                source = exported.constants.get(target)
            if (
                isinstance(source, torch.Tensor)
                and source.dtype is torch.int64
                and target.endswith("num_batches_tracked")
                and len(node.users) == 0
            ):
                # Some export versions lift this BatchNorm bookkeeping buffer
                # even though eval inference does not read it. It is safe to
                # omit only while the FX graph proves it has no users.
                continue
            array = _constant_array(torch, source, node.name)
            value_kind = {
                "PARAMETER": FloatValueKind.PARAMETER,
                "BUFFER": FloatValueKind.BUFFER,
                "CONSTANT_TENSOR": FloatValueKind.CONSTANT,
            }[kind]
            values[node.name] = FloatValue(
                node.name,
                tuple(array.shape),
                FloatDType.FLOAT32,
                _layout(tuple(array.shape)),
                value_kind,
                target,
            )
            constants[node.name] = array
            continue
        if node.op == "get_attr":
            target = str(node.target)
            source = _fetch_attr(exported.graph_module, target)
            array = _constant_array(torch, source, node.name)
            values[node.name] = FloatValue(
                node.name,
                tuple(array.shape),
                FloatDType.FLOAT32,
                _layout(tuple(array.shape)),
                FloatValueKind.CONSTANT,
                target,
            )
            constants[node.name] = array
            continue
        if node.op == "call_function":
            values[node.name] = _activation_value(torch, node)
            if str(node.target) == "aten.batch_norm.default":
                _fold_batch_norm(node, values, constants, ops, reserved_value_names)
                continue
            if str(node.target) in ("aten.dropout.default", "aten.dropout_.default"):
                _remove_eval_dropout(node, values)
                continue
            if str(node.target) == "aten.cat.default":
                sequence = _arg(node, 0, None)
                if isinstance(sequence, (tuple, list)) and len(sequence) == 1:
                    input_name = _tensor_name(sequence[0], values, f"{node.name} input")
                    if values[node.name].shape != values[input_name].shape:
                        raise CompileError(
                            f"{node.name}: single-input Concat must preserve input shape"
                        )
                    # torch.cat([x]) is an allocation in eager mode, but its
                    # tensor values are identical.  The immutable AOT graph can
                    # safely canonicalize it to an SSA alias and avoid emitting
                    # a copy-only operation.  DenseNet emits this pattern for
                    # the first feature list in each dense block.
                    values[node.name] = values[input_name]
                    continue
            ops.append(_extract_op(node, values, constants))
            continue
        if node.op == "output":
            output_node = node
            continue
        raise CompileError(f"{node.name}: unsupported torch.export node kind {node.op}")

    if len(graph_inputs) != 1:
        raise CompileError(f"P0 requires exactly one user input, captured {len(graph_inputs)}")
    if output_node is None:
        raise CompileError("torch.export graph has no output node")
    raw_outputs = output_node.args[0]
    if not isinstance(raw_outputs, (tuple, list)):
        raw_outputs = (raw_outputs,)
    if len(raw_outputs) != 1:
        raise CompileError(f"P0 requires exactly one tensor output, captured {len(raw_outputs)}")
    output_name = _tensor_name(raw_outputs[0], values, "graph output")
    if ops and output_name != ops[-1].outputs[0]:  # type: ignore[attr-defined]
        raise CompileError("P0 graph output must be produced by the final supported operation")
    if not ops and output_name not in graph_inputs:
        raise CompileError("operation-free P0 graph output must alias the user input")
    graph_name = name or type(model).__name__.lower()
    if not isinstance(graph_name, str) or not graph_name:
        raise CompileError("captured graph name must be non-empty")
    return FloatGraph(
        graph_name,
        {key: value for key, value in values.items() if key == value.name},
        constants,
        tuple(ops),  # type: ignore[arg-type]
        tuple(graph_inputs),
        (output_name,),
    )


capture = capture_torch_export


__all__ = ["ALLOWED_ATEN_TARGETS", "capture", "capture_torch_export"]
