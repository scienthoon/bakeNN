#!/usr/bin/env python3
"""Build a TFLite FlatBuffer with the exact BakeNN target-smoke INT8 tensors.

This intentionally uses only the generated TFLite schema package and
flatbuffers; installing the full TensorFlow Python package is unnecessary.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import flatbuffers
import numpy as np
import tflite

from bakenn.ir import PerAxisQParams, PerTensorQParams
from examples.targets.generate_smoke import smoke_graph


def _int32_vector(builder: flatbuffers.Builder, values: tuple[int, ...]) -> int:
    builder.StartVector(4, len(values), 4)
    for value in reversed(values):
        builder.PrependInt32(value)
    return builder.EndVector()


def _int64_vector(builder: flatbuffers.Builder, values: tuple[int, ...]) -> int:
    builder.StartVector(8, len(values), 8)
    for value in reversed(values):
        builder.PrependInt64(value)
    return builder.EndVector()


def _float32_vector(builder: flatbuffers.Builder, values: tuple[float, ...]) -> int:
    builder.StartVector(4, len(values), 4)
    for value in reversed(values):
        builder.PrependFloat32(value)
    return builder.EndVector()


def _offset_vector(builder: flatbuffers.Builder, offsets: tuple[int, ...]) -> int:
    builder.StartVector(4, len(offsets), 4)
    for offset in reversed(offsets):
        builder.PrependUOffsetTRelative(offset)
    return builder.EndVector()


def _quantization(builder: flatbuffers.Builder, qparams: object) -> int:
    if isinstance(qparams, PerTensorQParams):
        scales = (float(qparams.scale),)
        zero_points = (int(qparams.zero_point),)
        axis = 0
    elif isinstance(qparams, PerAxisQParams):
        scales = tuple(float(value) for value in qparams.scales)
        zero_points = tuple(int(value) for value in qparams.zero_points)
        axis = qparams.axis
    else:
        raise TypeError(f"unsupported qparams {type(qparams).__name__}")
    scale_vector = _float32_vector(builder, scales)
    zero_vector = _int64_vector(builder, zero_points)
    tflite.QuantizationParametersStart(builder)
    tflite.QuantizationParametersAddScale(builder, scale_vector)
    tflite.QuantizationParametersAddZeroPoint(builder, zero_vector)
    tflite.QuantizationParametersAddQuantizedDimension(builder, axis)
    return tflite.QuantizationParametersEnd(builder)


def _tensor(
    builder: flatbuffers.Builder,
    name: str,
    shape: tuple[int, ...],
    tensor_type: int,
    buffer_index: int,
    qparams: object,
) -> int:
    name_offset = builder.CreateString(name)
    shape_offset = _int32_vector(builder, shape)
    quantization = _quantization(builder, qparams)
    tflite.TensorStart(builder)
    tflite.TensorAddShape(builder, shape_offset)
    tflite.TensorAddType(builder, tensor_type)
    tflite.TensorAddBuffer(builder, buffer_index)
    tflite.TensorAddName(builder, name_offset)
    tflite.TensorAddQuantization(builder, quantization)
    return tflite.TensorEnd(builder)


def _buffer(builder: flatbuffers.Builder, value: np.ndarray | None) -> int:
    data = 0 if value is None else builder.CreateByteVector(value.tobytes(order="C"))
    tflite.BufferStart(builder)
    if data:
        tflite.BufferAddData(builder, data)
    return tflite.BufferEnd(builder)


def _fully_connected_options(builder: flatbuffers.Builder, activation: int) -> int:
    tflite.FullyConnectedOptionsStart(builder)
    tflite.FullyConnectedOptionsAddFusedActivationFunction(builder, activation)
    tflite.FullyConnectedOptionsAddWeightsFormat(
        builder, tflite.FullyConnectedOptionsWeightsFormat.DEFAULT
    )
    tflite.FullyConnectedOptionsAddKeepNumDims(builder, False)
    tflite.FullyConnectedOptionsAddAsymmetricQuantizeInputs(builder, False)
    return tflite.FullyConnectedOptionsEnd(builder)


def _fully_connected_operator(
    builder: flatbuffers.Builder,
    inputs: tuple[int, int, int],
    output: int,
    activation: int,
) -> int:
    input_vector = _int32_vector(builder, inputs)
    output_vector = _int32_vector(builder, (output,))
    options = _fully_connected_options(builder, activation)
    tflite.OperatorStart(builder)
    tflite.OperatorAddOpcodeIndex(builder, 0)
    tflite.OperatorAddInputs(builder, input_vector)
    tflite.OperatorAddOutputs(builder, output_vector)
    tflite.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.FullyConnectedOptions)
    tflite.OperatorAddBuiltinOptions(builder, options)
    return tflite.OperatorEnd(builder)


def _conv2d_options(builder: flatbuffers.Builder, activation: int) -> int:
    from tflite.Conv2DOptions import (
        Conv2DOptionsAddDilationHFactor,
        Conv2DOptionsAddDilationWFactor,
        Conv2DOptionsAddFusedActivationFunction,
        Conv2DOptionsAddPadding,
        Conv2DOptionsAddStrideH,
        Conv2DOptionsAddStrideW,
        Conv2DOptionsEnd,
        Conv2DOptionsStart,
    )

    Conv2DOptionsStart(builder)
    Conv2DOptionsAddPadding(builder, tflite.Padding.SAME)
    Conv2DOptionsAddStrideH(builder, 1)
    Conv2DOptionsAddStrideW(builder, 1)
    Conv2DOptionsAddFusedActivationFunction(builder, activation)
    Conv2DOptionsAddDilationHFactor(builder, 1)
    Conv2DOptionsAddDilationWFactor(builder, 1)
    return Conv2DOptionsEnd(builder)


def _conv2d_operator(
    builder: flatbuffers.Builder,
    inputs: tuple[int, int, int],
    output: int,
    activation: int,
) -> int:
    from tflite.Operator import (
        OperatorAddBuiltinOptions,
        OperatorAddBuiltinOptionsType,
        OperatorAddInputs,
        OperatorAddOpcodeIndex,
        OperatorAddOutputs,
        OperatorStart,
        OperatorEnd,
    )

    input_vector = _int32_vector(builder, inputs)
    output_vector = _int32_vector(builder, (output,))
    options = _conv2d_options(builder, activation)
    OperatorStart(builder)
    OperatorAddOpcodeIndex(builder, 0)
    OperatorAddInputs(builder, input_vector)
    OperatorAddOutputs(builder, output_vector)
    OperatorAddBuiltinOptionsType(
        builder, tflite.BuiltinOptions.Conv2DOptions
    )
    OperatorAddBuiltinOptions(builder, options)
    return OperatorEnd(builder)


def _model(
    builder: flatbuffers.Builder,
    *,
    tensors: tuple[int, ...],
    input_index: int,
    output_index: int,
    operators: tuple[int, ...],
    opcode: int,
    description: str,
    graph_name: str | None = None,
    buffers: tuple[int, ...],
) -> bytes:
    tensor_vector = _offset_vector(builder, tensors)
    input_vector = _int32_vector(builder, (input_index,))
    output_vector = _int32_vector(builder, (output_index,))
    operator_vector = _offset_vector(builder, operators)
    subgraph_name = builder.CreateString(graph_name or description)
    tflite.SubGraphStart(builder)
    tflite.SubGraphAddTensors(builder, tensor_vector)
    tflite.SubGraphAddInputs(builder, input_vector)
    tflite.SubGraphAddOutputs(builder, output_vector)
    tflite.SubGraphAddOperators(builder, operator_vector)
    tflite.SubGraphAddName(builder, subgraph_name)
    subgraph = tflite.SubGraphEnd(builder)

    opcode_vector = _offset_vector(builder, (opcode,))
    subgraph_vector = _offset_vector(builder, (subgraph,))
    buffer_vector = _offset_vector(builder, buffers)
    description_offset = builder.CreateString(description)
    tflite.ModelStart(builder)
    tflite.ModelAddVersion(builder, 3)
    tflite.ModelAddOperatorCodes(builder, opcode_vector)
    tflite.ModelAddSubgraphs(builder, subgraph_vector)
    tflite.ModelAddDescription(builder, description_offset)
    tflite.ModelAddBuffers(builder, buffer_vector)
    model = tflite.ModelEnd(builder)
    builder.Finish(model, file_identifier=b"TFL3")
    return bytes(builder.Output())


def build_model() -> bytes:
    graph = smoke_graph()
    values = graph.values
    constants = graph.constants
    builder = flatbuffers.Builder(16 * 1024)

    buffers = (
        _buffer(builder, None),
        _buffer(builder, constants["hidden.weight"]),
        _buffer(builder, constants["hidden.bias"]),
        _buffer(builder, constants["output.weight"]),
        _buffer(builder, constants["output.bias"]),
    )
    tensors = (
        _tensor(builder, "input", (1, 32), tflite.TensorType.INT8, 0, values["input"].qparams),
        _tensor(
            builder,
            "hidden.weight",
            (16, 32),
            tflite.TensorType.INT8,
            1,
            values["hidden.weight"].qparams,
        ),
        _tensor(
            builder,
            "hidden.bias",
            (16,),
            tflite.TensorType.INT32,
            2,
            values["hidden.bias"].qparams,
        ),
        _tensor(
            builder,
            "hidden.output",
            (1, 16),
            tflite.TensorType.INT8,
            0,
            values["hidden.output"].qparams,
        ),
        _tensor(
            builder,
            "output.weight",
            (4, 16),
            tflite.TensorType.INT8,
            3,
            values["output.weight"].qparams,
        ),
        _tensor(
            builder,
            "output.bias",
            (4,),
            tflite.TensorType.INT32,
            4,
            values["output.bias"].qparams,
        ),
        _tensor(
            builder,
            "output",
            (1, 4),
            tflite.TensorType.INT8,
            0,
            values["output.output"].qparams,
        ),
    )
    operators = (
        _fully_connected_operator(
            builder, (0, 1, 2), 3, tflite.ActivationFunctionType.RELU
        ),
        _fully_connected_operator(
            builder, (3, 4, 5), 6, tflite.ActivationFunctionType.NONE
        ),
    )

    # Keep the original construction order stable: the generated FlatBuffer is
    # part of the benchmark provenance and its bytes should not churn when a
    # second model family is added.
    tensor_vector = _offset_vector(builder, tensors)
    input_vector = _int32_vector(builder, (0,))
    output_vector = _int32_vector(builder, (6,))
    operator_vector = _offset_vector(builder, operators)
    subgraph_name = builder.CreateString("target_smoke")
    tflite.SubGraphStart(builder)
    tflite.SubGraphAddTensors(builder, tensor_vector)
    tflite.SubGraphAddInputs(builder, input_vector)
    tflite.SubGraphAddOutputs(builder, output_vector)
    tflite.SubGraphAddOperators(builder, operator_vector)
    tflite.SubGraphAddName(builder, subgraph_name)
    subgraph = tflite.SubGraphEnd(builder)

    tflite.OperatorCodeStart(builder)
    tflite.OperatorCodeAddDeprecatedBuiltinCode(builder, tflite.BuiltinOperator.FULLY_CONNECTED)
    tflite.OperatorCodeAddBuiltinCode(builder, tflite.BuiltinOperator.FULLY_CONNECTED)
    tflite.OperatorCodeAddVersion(builder, 4)
    opcode = tflite.OperatorCodeEnd(builder)
    opcode_vector = _offset_vector(builder, (opcode,))
    subgraph_vector = _offset_vector(builder, (subgraph,))
    buffer_vector = _offset_vector(builder, buffers)
    description = builder.CreateString("BakeNN target-smoke TFLM comparison")
    tflite.ModelStart(builder)
    tflite.ModelAddVersion(builder, 3)
    tflite.ModelAddOperatorCodes(builder, opcode_vector)
    tflite.ModelAddSubgraphs(builder, subgraph_vector)
    tflite.ModelAddDescription(builder, description)
    tflite.ModelAddBuffers(builder, buffer_vector)
    model = tflite.ModelEnd(builder)
    builder.Finish(model, file_identifier=b"TFL3")
    return bytes(builder.Output())


def build_cmsis_fc_model() -> bytes:
    """Build the exact per-tensor-weight MLP used by the direct CMSIS test."""

    from benchmarks.tflm_compare.model_fixtures import cmsis_mlp_graph

    graph = cmsis_mlp_graph()
    values = graph.values
    constants = graph.constants
    builder = flatbuffers.Builder(16 * 1024)
    buffers = (
        _buffer(builder, None),
        _buffer(builder, constants["weight_0"]),
        _buffer(builder, constants["bias_0"]),
        _buffer(builder, constants["weight_1"]),
        _buffer(builder, constants["bias_1"]),
    )
    tensor_names = (
        ("input", 0, tflite.TensorType.INT8),
        ("weight_0", 1, tflite.TensorType.INT8),
        ("bias_0", 2, tflite.TensorType.INT32),
        ("activation_0", 0, tflite.TensorType.INT8),
        ("weight_1", 3, tflite.TensorType.INT8),
        ("bias_1", 4, tflite.TensorType.INT32),
        ("activation_1", 0, tflite.TensorType.INT8),
    )
    tensors = tuple(
        _tensor(
            builder,
            name,
            values[name].shape,
            tensor_type,
            buffer_index,
            values[name].qparams,
        )
        for name, buffer_index, tensor_type in tensor_names
    )
    operators = (
        _fully_connected_operator(
            builder, (0, 1, 2), 3, tflite.ActivationFunctionType.RELU
        ),
        _fully_connected_operator(
            builder, (3, 4, 5), 6, tflite.ActivationFunctionType.NONE
        ),
    )
    tflite.OperatorCodeStart(builder)
    tflite.OperatorCodeAddDeprecatedBuiltinCode(
        builder, tflite.BuiltinOperator.FULLY_CONNECTED
    )
    tflite.OperatorCodeAddBuiltinCode(builder, tflite.BuiltinOperator.FULLY_CONNECTED)
    tflite.OperatorCodeAddVersion(builder, 4)
    opcode = tflite.OperatorCodeEnd(builder)
    return _model(
        builder,
        tensors=tensors,
        input_index=0,
        output_index=6,
        operators=operators,
        opcode=opcode,
        description="BakeNN direct CMSIS-NN FC comparison",
        graph_name=graph.name,
        buffers=buffers,
    )


def build_conv_model() -> bytes:
    """Build the single 3x3 Conv2D from the deterministic tiny-CNN fixture."""

    from tests.p0.model_fixtures import tiny_cnn_graph

    graph = tiny_cnn_graph()
    values = graph.values
    constants = graph.constants
    builder = flatbuffers.Builder(16 * 1024)
    buffers = (
        _buffer(builder, None),
        _buffer(builder, constants["conv.weight"]),
        _buffer(builder, constants["conv.bias"]),
    )
    tensors = (
        _tensor(builder, "input", values["input"].shape, tflite.TensorType.INT8, 0, values["input"].qparams),
        _tensor(builder, "conv.weight", values["conv.weight"].shape, tflite.TensorType.INT8, 1, values["conv.weight"].qparams),
        _tensor(builder, "conv.bias", values["conv.bias"].shape, tflite.TensorType.INT32, 2, values["conv.bias"].qparams),
        _tensor(builder, "conv.output", values["conv.output"].shape, tflite.TensorType.INT8, 0, values["conv.output"].qparams),
    )
    operator = _conv2d_operator(
        builder,
        (0, 1, 2),
        3,
        tflite.ActivationFunctionType.RELU,
    )
    tflite.OperatorCodeStart(builder)
    tflite.OperatorCodeAddDeprecatedBuiltinCode(builder, tflite.BuiltinOperator.CONV_2D)
    tflite.OperatorCodeAddBuiltinCode(builder, tflite.BuiltinOperator.CONV_2D)
    # Zephyr 2.7's pinned TFLM (2021) accepts Conv2D version 2; the tensor and
    # quantization contract used here does not need a newer opcode revision.
    tflite.OperatorCodeAddVersion(builder, 2)
    opcode = tflite.OperatorCodeEnd(builder)
    return _model(
        builder,
        tensors=tensors,
        input_index=0,
        output_index=3,
        operators=(operator,),
        opcode=opcode,
        description="BakeNN tiny-CNN Conv2D TFLM comparison",
        buffers=buffers,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--c-array-output", type=Path)
    parser.add_argument(
        "--model", choices=("mlp", "cmsis-mlp", "conv"), default="mlp"
    )
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    if arguments.model == "mlp":
        data = build_model()
    elif arguments.model == "cmsis-mlp":
        data = build_cmsis_fc_model()
    else:
        data = build_conv_model()
    arguments.output.write_bytes(data)
    parsed = tflite.Model.GetRootAsModel(data, 0)
    if parsed.Version() != 3 or parsed.SubgraphsLength() != 1:
        raise RuntimeError("generated TFLite model failed schema self-check")
    if arguments.c_array_output is not None:
        arguments.c_array_output.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for start in range(0, len(data), 12):
            rows.append(
                "    " + ", ".join(f"0x{value:02x}" for value in data[start : start + 12])
            )
        input_count, output_count, arena_size = (
            (32, 4, 1024)
            if arguments.model in ("mlp", "cmsis-mlp")
            else (16, 32, 2048)
        )
        arguments.c_array_output.write_text(
            '#include "model_data.h"\n\n'
            "alignas(8) const unsigned char tflm_model_data[] = {\n"
            + ",\n".join(rows)
            + "\n};\n"
            + f"const unsigned int tflm_model_data_len = {len(data)}u;\n"
            + f"const unsigned int tflm_model_input_count = {input_count}u;\n"
            + f"const unsigned int tflm_model_output_count = {output_count}u;\n"
            + f"const unsigned int tflm_model_arena_size = {arena_size}u;\n",
            encoding="utf-8",
        )
    print(f"{arguments.output} ({len(data)} bytes)")


if __name__ == "__main__":
    sys.exit(main())
