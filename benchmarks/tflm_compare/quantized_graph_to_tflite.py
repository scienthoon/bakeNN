#!/usr/bin/env python3
"""Serialize the BakeNN P0 CNN subset as a fully-quantized TFLite model.

This adapter exists for apples-to-apples TFLM benchmarks.  It does not make
TFLite a BakeNN frontend or runtime dependency: the generated BakeNN C path
does not import or link any code from this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import flatbuffers
import numpy as np
import tflite

from bakenn.errors import CompileError
from bakenn.ir import (
    AddOp,
    AveragePool2DOp,
    Conv2DOp,
    DType,
    DepthwiseConv2DOp,
    FlattenOp,
    LinearOp,
    PerAxisQParams,
    PerTensorQParams,
    QuantizedGraph,
    ReshapeOp,
    verify_graph,
)


@dataclass(frozen=True)
class TFLiteExport:
    data: bytes
    operator_counts: dict[str, int]


def _int32_vector(builder: flatbuffers.Builder, values: tuple[int, ...]) -> int:
    builder.StartVector(4, len(values), 4)
    for value in reversed(values):
        builder.PrependInt32(int(value))
    return builder.EndVector()


def _int64_vector(builder: flatbuffers.Builder, values: tuple[int, ...]) -> int:
    builder.StartVector(8, len(values), 8)
    for value in reversed(values):
        builder.PrependInt64(int(value))
    return builder.EndVector()


def _float32_vector(builder: flatbuffers.Builder, values: tuple[float, ...]) -> int:
    builder.StartVector(4, len(values), 4)
    for value in reversed(values):
        builder.PrependFloat32(float(value))
    return builder.EndVector()


def _offset_vector(builder: flatbuffers.Builder, offsets: tuple[int, ...]) -> int:
    builder.StartVector(4, len(offsets), 4)
    for offset in reversed(offsets):
        builder.PrependUOffsetTRelative(offset)
    return builder.EndVector()


def _quantization(
    builder: flatbuffers.Builder,
    qparams: PerTensorQParams | PerAxisQParams,
    *,
    axis_override: int | None = None,
) -> int:
    if isinstance(qparams, PerTensorQParams):
        scales = (qparams.scale,)
        zero_points = (qparams.zero_point,)
        axis = 0
    else:
        scales = qparams.scales
        zero_points = qparams.zero_points
        axis = qparams.axis if axis_override is None else axis_override
    scale_vector = _float32_vector(builder, tuple(scales))
    zero_vector = _int64_vector(builder, tuple(zero_points))
    tflite.QuantizationParametersStart(builder)
    tflite.QuantizationParametersAddScale(builder, scale_vector)
    tflite.QuantizationParametersAddZeroPoint(builder, zero_vector)
    tflite.QuantizationParametersAddQuantizedDimension(builder, axis)
    return tflite.QuantizationParametersEnd(builder)


def _tensor(
    builder: flatbuffers.Builder,
    *,
    name: str,
    shape: tuple[int, ...],
    tensor_type: int,
    buffer_index: int,
    qparams: PerTensorQParams | PerAxisQParams | None,
    axis_override: int | None = None,
) -> int:
    name_offset = builder.CreateString(name)
    shape_offset = _int32_vector(builder, shape)
    quantization = (
        0
        if qparams is None
        else _quantization(builder, qparams, axis_override=axis_override)
    )
    tflite.TensorStart(builder)
    tflite.TensorAddShape(builder, shape_offset)
    tflite.TensorAddType(builder, tensor_type)
    tflite.TensorAddBuffer(builder, buffer_index)
    tflite.TensorAddName(builder, name_offset)
    if quantization:
        tflite.TensorAddQuantization(builder, quantization)
    return tflite.TensorEnd(builder)


def _buffer(builder: flatbuffers.Builder, value: np.ndarray | None) -> int:
    data = 0 if value is None else builder.CreateByteVector(value.tobytes(order="C"))
    tflite.BufferStart(builder)
    if data:
        tflite.BufferAddData(builder, data)
    return tflite.BufferEnd(builder)


def _padding(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    kernel: tuple[int, int],
    stride: tuple[int, int],
    dilation: tuple[int, int],
    padding: tuple[int, int, int, int],
) -> int | None:
    input_h, input_w = input_shape[1:3]
    output_h, output_w = output_shape[1:3]
    effective_h = (kernel[0] - 1) * dilation[0] + 1
    effective_w = (kernel[1] - 1) * dilation[1] + 1
    same_h = max(0, (output_h - 1) * stride[0] + effective_h - input_h)
    same_w = max(0, (output_w - 1) * stride[1] + effective_w - input_w)
    expected_same = (
        same_h // 2,
        same_h - same_h // 2,
        same_w // 2,
        same_w - same_w // 2,
    )
    if padding == expected_same and (
        output_h == math.ceil(input_h / stride[0])
        and output_w == math.ceil(input_w / stride[1])
    ):
        return tflite.Padding.SAME
    valid_h = (input_h - effective_h) // stride[0] + 1
    valid_w = (input_w - effective_w) // stride[1] + 1
    if padding == (0, 0, 0, 0) and (output_h, output_w) == (valid_h, valid_w):
        return tflite.Padding.VALID
    return None


def _rounded_code(real: float, qparams: PerTensorQParams) -> int:
    scaled = real / qparams.scale
    rounded = math.floor(scaled + 0.5) if scaled >= 0.0 else math.ceil(scaled - 0.5)
    return max(-128, min(127, rounded + qparams.zero_point))


def _activation(op: object, output_qparams: object) -> int:
    low = int(getattr(op, "activation_min"))
    high = int(getattr(op, "activation_max"))
    if (low, high) == (-128, 127):
        return tflite.ActivationFunctionType.NONE
    if not isinstance(output_qparams, PerTensorQParams):
        raise CompileError("fused activation requires per-tensor output qparams")
    zero_code = _rounded_code(0.0, output_qparams)
    if (low, high) == (zero_code, 127):
        return tflite.ActivationFunctionType.RELU
    six_code = _rounded_code(6.0, output_qparams)
    if (low, high) == (zero_code, six_code):
        return tflite.ActivationFunctionType.RELU6
    raise CompileError("BakeNN raw-code clamp is not a TFLite fused activation")


def _operator(
    builder: flatbuffers.Builder,
    opcode_index: int,
    inputs: tuple[int, ...],
    output: int,
    options_type: int,
    options: int,
) -> int:
    inputs_vector = _int32_vector(builder, inputs)
    outputs_vector = _int32_vector(builder, (output,))
    tflite.OperatorStart(builder)
    tflite.OperatorAddOpcodeIndex(builder, opcode_index)
    tflite.OperatorAddInputs(builder, inputs_vector)
    tflite.OperatorAddOutputs(builder, outputs_vector)
    tflite.OperatorAddBuiltinOptionsType(builder, options_type)
    tflite.OperatorAddBuiltinOptions(builder, options)
    return tflite.OperatorEnd(builder)


def _opcode(builder: flatbuffers.Builder, builtin: int, version: int) -> int:
    tflite.OperatorCodeStart(builder)
    tflite.OperatorCodeAddDeprecatedBuiltinCode(builder, builtin)
    tflite.OperatorCodeAddBuiltinCode(builder, builtin)
    tflite.OperatorCodeAddVersion(builder, version)
    return tflite.OperatorCodeEnd(builder)


def export_quantized_graph(graph: QuantizedGraph) -> TFLiteExport:
    """Return a TFLite FlatBuffer for the strict benchmarkable CNN subset."""

    verify_graph(graph)
    if len(graph.inputs) != 1 or len(graph.outputs) != 1:
        raise CompileError("TFLite comparison adapter requires one input and one output")

    supported = (
        Conv2DOp,
        DepthwiseConv2DOp,
        AddOp,
        AveragePool2DOp,
        FlattenOp,
        ReshapeOp,
        LinearOp,
    )
    unsupported = [type(op).__name__ for op in graph.ops if not isinstance(op, supported)]
    if unsupported:
        raise CompileError(f"unsupported TFLite comparison op: {unsupported[0]}")

    builder = flatbuffers.Builder(2 * 1024 * 1024)
    value_names = list(graph.values)
    tensor_indices = {name: index for index, name in enumerate(value_names)}
    synthetic: dict[
        str,
        tuple[
            tuple[int, ...],
            int,
            PerTensorQParams | PerAxisQParams | None,
            np.ndarray | None,
        ],
    ] = {}

    reshape_names: dict[str, str] = {}
    explicit_pads: dict[str, tuple[str, str, str]] = {}
    for op in graph.ops:
        if isinstance(op, (FlattenOp, ReshapeOp)):
            name = f"{op.name}.tflite_shape"
            if name in tensor_indices:
                raise CompileError(f"synthetic TFLite tensor name collision: {name}")
            reshape_names[op.name] = name
            tensor_indices[name] = len(value_names)
            value_names.append(name)
            shape = np.asarray(graph.values[op.output].shape, dtype=np.int32)
            synthetic[name] = (tuple(shape.shape), tflite.TensorType.INT32, None, shape)
        if isinstance(op, (Conv2DOp, DepthwiseConv2DOp)):
            input_type = graph.values[op.input]
            output_type = graph.values[op.output]
            weight_shape = graph.values[op.weight].shape
            kernel = weight_shape[1:3] if isinstance(op, Conv2DOp) else weight_shape[:2]
            representable = _padding(
                input_type.shape,
                output_type.shape,
                kernel,
                op.stride,
                op.dilation,
                op.padding,
            )
            if representable is None:
                top, bottom, left, right = op.padding
                padded_shape = (
                    input_type.shape[0],
                    input_type.shape[1] + top + bottom,
                    input_type.shape[2] + left + right,
                    input_type.shape[3],
                )
                effective_h = (kernel[0] - 1) * op.dilation[0] + 1
                effective_w = (kernel[1] - 1) * op.dilation[1] + 1
                expected_output = (
                    (padded_shape[1] - effective_h) // op.stride[0] + 1,
                    (padded_shape[2] - effective_w) // op.stride[1] + 1,
                )
                if expected_output != output_type.shape[1:3]:
                    raise CompileError(
                        f"{op.name}: explicit padding cannot preserve the output shape"
                    )
                output_name = f"{op.name}.tflite_pad_output"
                paddings_name = f"{op.name}.tflite_paddings"
                pad_value_name = f"{op.name}.tflite_pad_value"
                for name in (output_name, paddings_name, pad_value_name):
                    if name in tensor_indices:
                        raise CompileError(f"synthetic TFLite tensor name collision: {name}")
                    tensor_indices[name] = len(value_names)
                    value_names.append(name)
                paddings = np.asarray(
                    ((0, 0), (top, bottom), (left, right), (0, 0)), dtype=np.int32
                )
                synthetic[output_name] = (
                    padded_shape,
                    tflite.TensorType.INT8,
                    input_type.qparams,
                    None,
                )
                synthetic[paddings_name] = (
                    tuple(paddings.shape),
                    tflite.TensorType.INT32,
                    None,
                    paddings,
                )
                if not isinstance(input_type.qparams, PerTensorQParams):
                    raise CompileError("explicit activation padding requires per-tensor qparams")
                synthetic[pad_value_name] = (
                    (1,),
                    tflite.TensorType.INT8,
                    input_type.qparams,
                    np.asarray((input_type.qparams.zero_point,), dtype=np.int8),
                )
                explicit_pads[op.name] = (
                    output_name,
                    paddings_name,
                    pad_value_name,
                )

    buffer_values: list[np.ndarray | None] = [None]
    buffer_indices: dict[str, int] = {}
    depthwise_weights = {
        op.weight for op in graph.ops if isinstance(op, DepthwiseConv2DOp)
    }
    for name, value in graph.constants.items():
        stored = value.reshape((1, *value.shape)) if name in depthwise_weights else value
        buffer_indices[name] = len(buffer_values)
        buffer_values.append(np.ascontiguousarray(stored))
    for name, (_, _, _, stored) in synthetic.items():
        if stored is not None:
            buffer_indices[name] = len(buffer_values)
            buffer_values.append(stored)
    buffers = tuple(_buffer(builder, value) for value in buffer_values)

    tensors: list[int] = []
    for name in value_names:
        if name in graph.values:
            value_type = graph.values[name]
            shape = value_type.shape
            axis_override = None
            if name in depthwise_weights:
                shape = (1, *shape)
                axis_override = 3
            tensor_type = (
                tflite.TensorType.INT8
                if value_type.dtype is DType.INT8
                else tflite.TensorType.INT32
            )
            tensors.append(
                _tensor(
                    builder,
                    name=name,
                    shape=shape,
                    tensor_type=tensor_type,
                    buffer_index=buffer_indices.get(name, 0),
                    qparams=value_type.qparams,
                    axis_override=axis_override,
                )
            )
        else:
            shape, tensor_type, qparams, _ = synthetic[name]
            tensors.append(
                _tensor(
                    builder,
                    name=name,
                    shape=shape,
                    tensor_type=tensor_type,
                    buffer_index=buffer_indices.get(name, 0),
                    qparams=qparams,
                )
            )

    opcode_specs = (
        ("CONV_2D", tflite.BuiltinOperator.CONV_2D, 3),
        ("DEPTHWISE_CONV_2D", tflite.BuiltinOperator.DEPTHWISE_CONV_2D, 3),
        ("ADD", tflite.BuiltinOperator.ADD, 2),
        ("AVERAGE_POOL_2D", tflite.BuiltinOperator.AVERAGE_POOL_2D, 2),
        ("RESHAPE", tflite.BuiltinOperator.RESHAPE, 1),
        ("FULLY_CONNECTED", tflite.BuiltinOperator.FULLY_CONNECTED, 4),
        ("PADV2", tflite.BuiltinOperator.PADV2, 1),
    )
    opcode_indices = {name: index for index, (name, _, _) in enumerate(opcode_specs)}

    operators: list[int] = []
    counts: dict[str, int] = {}
    for op in graph.ops:
        output_type = graph.values[op.outputs[0]]
        activation = _activation(op, output_type.qparams) if hasattr(op, "activation_min") else 0
        explicit_pad = explicit_pads.get(op.name)
        if explicit_pad is not None:
            padded_name, paddings_name, pad_value_name = explicit_pad
            tflite.PadOptionsStart(builder)
            pad_options = tflite.PadOptionsEnd(builder)
            operators.append(
                _operator(
                    builder,
                    opcode_indices["PADV2"],
                    (
                        tensor_indices[op.input],
                        tensor_indices[paddings_name],
                        tensor_indices[pad_value_name],
                    ),
                    tensor_indices[padded_name],
                    tflite.BuiltinOptions.PadOptions,
                    pad_options,
                )
            )
            counts["PADV2"] = counts.get("PADV2", 0) + 1
        if isinstance(op, Conv2DOp):
            weight_shape = graph.values[op.weight].shape
            padding = _padding(
                graph.values[op.input].shape,
                output_type.shape,
                weight_shape[1:3],
                op.stride,
                op.dilation,
                op.padding,
            )
            if explicit_pad is not None:
                padding = tflite.Padding.VALID
            assert padding is not None
            tflite.Conv2DOptionsStart(builder)
            tflite.Conv2DOptionsAddPadding(builder, padding)
            tflite.Conv2DOptionsAddStrideW(builder, op.stride[1])
            tflite.Conv2DOptionsAddStrideH(builder, op.stride[0])
            tflite.Conv2DOptionsAddFusedActivationFunction(builder, activation)
            tflite.Conv2DOptionsAddDilationWFactor(builder, op.dilation[1])
            tflite.Conv2DOptionsAddDilationHFactor(builder, op.dilation[0])
            options = tflite.Conv2DOptionsEnd(builder)
            kind = "CONV_2D"
            input_name = explicit_pad[0] if explicit_pad is not None else op.input
            inputs = (
                tensor_indices[input_name],
                tensor_indices[op.weight],
                tensor_indices[op.bias],
            )
            options_type = tflite.BuiltinOptions.Conv2DOptions
        elif isinstance(op, DepthwiseConv2DOp):
            weight_shape = graph.values[op.weight].shape
            padding = _padding(
                graph.values[op.input].shape,
                output_type.shape,
                weight_shape[:2],
                op.stride,
                op.dilation,
                op.padding,
            )
            if explicit_pad is not None:
                padding = tflite.Padding.VALID
            assert padding is not None
            tflite.DepthwiseConv2DOptionsStart(builder)
            tflite.DepthwiseConv2DOptionsAddPadding(builder, padding)
            tflite.DepthwiseConv2DOptionsAddStrideW(builder, op.stride[1])
            tflite.DepthwiseConv2DOptionsAddStrideH(builder, op.stride[0])
            tflite.DepthwiseConv2DOptionsAddDepthMultiplier(builder, op.depth_multiplier)
            tflite.DepthwiseConv2DOptionsAddFusedActivationFunction(builder, activation)
            tflite.DepthwiseConv2DOptionsAddDilationWFactor(builder, op.dilation[1])
            tflite.DepthwiseConv2DOptionsAddDilationHFactor(builder, op.dilation[0])
            options = tflite.DepthwiseConv2DOptionsEnd(builder)
            kind = "DEPTHWISE_CONV_2D"
            input_name = explicit_pad[0] if explicit_pad is not None else op.input
            inputs = (
                tensor_indices[input_name],
                tensor_indices[op.weight],
                tensor_indices[op.bias],
            )
            options_type = tflite.BuiltinOptions.DepthwiseConv2DOptions
        elif isinstance(op, AddOp):
            tflite.AddOptionsStart(builder)
            tflite.AddOptionsAddFusedActivationFunction(builder, activation)
            tflite.AddOptionsAddPotScaleInt16(builder, True)
            options = tflite.AddOptionsEnd(builder)
            kind = "ADD"
            inputs = tuple(tensor_indices[name] for name in op.inputs)
            options_type = tflite.BuiltinOptions.AddOptions
        elif isinstance(op, AveragePool2DOp):
            padding = _padding(
                graph.values[op.input].shape,
                output_type.shape,
                op.kernel,
                op.stride,
                (1, 1),
                op.padding,
            )
            if padding is None:
                raise CompileError(f"{op.name}: pool padding is not TFLite SAME/VALID")
            tflite.Pool2DOptionsStart(builder)
            tflite.Pool2DOptionsAddPadding(builder, padding)
            tflite.Pool2DOptionsAddStrideW(builder, op.stride[1])
            tflite.Pool2DOptionsAddStrideH(builder, op.stride[0])
            tflite.Pool2DOptionsAddFilterWidth(builder, op.kernel[1])
            tflite.Pool2DOptionsAddFilterHeight(builder, op.kernel[0])
            tflite.Pool2DOptionsAddFusedActivationFunction(builder, activation)
            options = tflite.Pool2DOptionsEnd(builder)
            kind = "AVERAGE_POOL_2D"
            inputs = (tensor_indices[op.input],)
            options_type = tflite.BuiltinOptions.Pool2DOptions
        elif isinstance(op, (FlattenOp, ReshapeOp)):
            shape = output_type.shape
            shape_vector = _int32_vector(builder, shape)
            tflite.ReshapeOptionsStart(builder)
            tflite.ReshapeOptionsAddNewShape(builder, shape_vector)
            options = tflite.ReshapeOptionsEnd(builder)
            kind = "RESHAPE"
            inputs = (tensor_indices[op.input], tensor_indices[reshape_names[op.name]])
            options_type = tflite.BuiltinOptions.ReshapeOptions
        elif isinstance(op, LinearOp):
            tflite.FullyConnectedOptionsStart(builder)
            tflite.FullyConnectedOptionsAddFusedActivationFunction(builder, activation)
            tflite.FullyConnectedOptionsAddWeightsFormat(
                builder, tflite.FullyConnectedOptionsWeightsFormat.DEFAULT
            )
            tflite.FullyConnectedOptionsAddKeepNumDims(builder, False)
            tflite.FullyConnectedOptionsAddAsymmetricQuantizeInputs(builder, False)
            options = tflite.FullyConnectedOptionsEnd(builder)
            kind = "FULLY_CONNECTED"
            inputs = tuple(tensor_indices[name] for name in op.inputs)
            options_type = tflite.BuiltinOptions.FullyConnectedOptions
        else:  # pragma: no cover - guarded above
            raise AssertionError(type(op).__name__)
        counts[kind] = counts.get(kind, 0) + 1
        operators.append(
            _operator(
                builder,
                opcode_indices[kind],
                inputs,
                tensor_indices[op.outputs[0]],
                options_type,
                options,
            )
        )

    tensor_vector = _offset_vector(builder, tuple(tensors))
    input_vector = _int32_vector(builder, (tensor_indices[graph.inputs[0]],))
    output_vector = _int32_vector(builder, (tensor_indices[graph.outputs[0]],))
    operator_vector = _offset_vector(builder, tuple(operators))
    graph_name = builder.CreateString(graph.name)
    tflite.SubGraphStart(builder)
    tflite.SubGraphAddTensors(builder, tensor_vector)
    tflite.SubGraphAddInputs(builder, input_vector)
    tflite.SubGraphAddOutputs(builder, output_vector)
    tflite.SubGraphAddOperators(builder, operator_vector)
    tflite.SubGraphAddName(builder, graph_name)
    subgraph = tflite.SubGraphEnd(builder)

    opcodes = tuple(_opcode(builder, builtin, version) for _, builtin, version in opcode_specs)
    opcode_vector = _offset_vector(builder, opcodes)
    subgraph_vector = _offset_vector(builder, (subgraph,))
    buffer_vector = _offset_vector(builder, buffers)
    description = builder.CreateString("BakeNN quantized-graph TFLM comparison")
    tflite.ModelStart(builder)
    tflite.ModelAddVersion(builder, 3)
    tflite.ModelAddOperatorCodes(builder, opcode_vector)
    tflite.ModelAddSubgraphs(builder, subgraph_vector)
    tflite.ModelAddDescription(builder, description)
    tflite.ModelAddBuffers(builder, buffer_vector)
    model = tflite.ModelEnd(builder)
    builder.Finish(model, file_identifier=b"TFL3")
    data = bytes(builder.Output())
    parsed = tflite.Model.GetRootAsModel(data, 0)
    if parsed.Version() != 3 or parsed.SubgraphsLength() != 1:
        raise RuntimeError("generated TFLite model failed schema self-check")
    return TFLiteExport(data=data, operator_counts=counts)


__all__ = ["TFLiteExport", "export_quantized_graph"]
