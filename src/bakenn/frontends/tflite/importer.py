"""Fail-closed import of a static fully-quantized TFLite subset.

The optional FlatBuffer/schema packages are loaded only when :func:`import_tflite`
is called.  Imported objects are converted immediately into BakeNN-owned IR;
neither the compiler core nor generated firmware retains a TFLite dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from bakenn.errors import BakeNNError, CompileError, Diagnostic
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
    Pad2DOp,
    PerAxisQParams,
    PerTensorQParams,
    QuantizedGraph,
    ReshapeOp,
    TensorType,
    verify_graph,
)
from bakenn.quantization.fixedpoint import round_half_away_from_zero
from bakenn.ir.types import normalize_scale_float32
from bakenn.ir.ops.pool import AVERAGE_POOL_PROFILE_TFLITE_RAW_V1


_SCHEMA_VERSION = 3
_FILE_IDENTIFIER = b"TFL3"


@dataclass(frozen=True)
class _Tensor:
    index: int
    name: str
    shape: tuple[int, ...]
    type_code: int
    buffer_index: int
    scales: tuple[float, ...]
    zero_points: tuple[int, ...]
    quantized_dimension: int
    data: bytes | None


@dataclass(frozen=True)
class _OperatorCode:
    builtin: int
    version: int
    custom_code: str | None


def _optional_dependencies() -> ModuleType:
    missing: list[str] = []
    for package in ("flatbuffers", "tflite"):
        try:
            importlib.import_module(package)
        except (ImportError, ModuleNotFoundError):
            missing.append(package)
    if missing:
        packages = ", ".join(missing)
        raise CompileError(
            "TFLite import requires optional host packages "
            f"{packages}; install BakeNN's TFLite extra or install "
            "'flatbuffers' and 'tflite' explicitly"
        )
    schema = importlib.import_module("tflite")
    required_schema_api = (
        ("Tensor", "VariantTensorsLength"),
        ("Buffer", "Offset"),
        ("Buffer", "Size"),
        ("Operator", "LargeCustomOptionsOffset"),
        ("Operator", "LargeCustomOptionsSize"),
        ("Operator", "BuiltinOptions2Type"),
        ("FullyConnectedOptions", "QuantizedBiasType"),
        ("Conv2DOptions", "QuantizedBiasType"),
    )
    unavailable = [
        f"{class_name}.{member}"
        for class_name, member in required_schema_api
        if not hasattr(getattr(schema, class_name, None), member)
    ]
    if unavailable:
        raise CompileError(
            "the installed 'tflite' schema package is too old for fail-closed "
            "import; install tflite>=2.18,<3 (missing schema API: "
            f"{', '.join(unavailable)})"
        )
    return schema


def _read_source(source: object) -> tuple[bytes, str | None]:
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source), None
    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            return path.read_bytes(), path.stem
        except OSError as error:
            raise CompileError(f"cannot read TFLite model {path}: {error}") from error
    raise CompileError("TFLite source must be a path or bytes-like object")


def _vector(value: object) -> tuple[int, ...]:
    if isinstance(value, (int, np.integer)) and int(value) == 0:
        return ()
    array = np.asarray(value)
    if array.size == 0:
        return ()
    return tuple(int(item) for item in array.reshape(-1))


def _float_vector(value: object) -> tuple[float, ...]:
    if isinstance(value, (int, np.integer)) and int(value) == 0:
        return ()
    array = np.asarray(value)
    if array.size == 0:
        return ()
    return tuple(float(item) for item in array.reshape(-1))


def _decode_text(value: object, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, bytes):
        try:
            decoded = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise CompileError(f"TFLite name is not valid UTF-8: {error}") from error
    else:
        decoded = str(value)
    return decoded if decoded else fallback


def _option(schema: ModuleType, operator: object, class_name: str, expected_type: int) -> Any:
    actual_type = int(operator.BuiltinOptionsType())  # type: ignore[attr-defined]
    if actual_type != expected_type:
        raise CompileError(
            f"TFLite {class_name} has builtin-options type {actual_type}, expected {expected_type}"
        )
    table = operator.BuiltinOptions()  # type: ignore[attr-defined]
    if table is None:
        raise CompileError(f"TFLite {class_name} is missing builtin options")
    options_class = getattr(schema, class_name)
    options = options_class()
    options.Init(table.Bytes, table.Pos)
    return options


def _normalize_shape_spec(
    spec: tuple[int, ...], input_shape: tuple[int, ...], output_shape: tuple[int, ...]
) -> tuple[int, ...]:
    if not spec:
        raise CompileError("TFLite Reshape requires a non-empty constant shape")
    if any(value == 0 or value < -1 for value in spec):
        raise CompileError("TFLite Reshape supports only positive dimensions and one -1")
    if sum(value == -1 for value in spec) > 1:
        raise CompileError("TFLite Reshape contains more than one inferred dimension")
    input_elements = math.prod(input_shape)
    known_elements = math.prod(value for value in spec if value != -1)
    resolved = list(spec)
    if -1 in resolved:
        if known_elements <= 0 or input_elements % known_elements:
            raise CompileError("TFLite Reshape inferred dimension is not integral")
        resolved[resolved.index(-1)] = input_elements // known_elements
    if math.prod(resolved) != input_elements or tuple(resolved) != output_shape:
        raise CompileError(
            f"TFLite Reshape constant resolves to {tuple(resolved)}, output is {output_shape}"
        )
    return tuple(resolved)


class _Importer:
    def __init__(self, schema: ModuleType, model: object, subgraph: object, graph_name: str):
        self.schema = schema
        self.model = model
        self.subgraph = subgraph
        self.graph_name = graph_name
        self.values: dict[str, TensorType] = {}
        self.constants: dict[str, np.ndarray] = {}
        self.ops: list[object] = []
        self.roles: dict[int, str] = {}
        self.produced: set[int] = set()
        self.input_indices = _vector(subgraph.InputsAsNumpy())
        self.output_indices = _vector(subgraph.OutputsAsNumpy())
        self.tensors = self._read_tensors()
        self.opcodes = self._read_opcodes()

    def _read_tensors(self) -> tuple[_Tensor, ...]:
        names: set[str] = set()
        records: list[_Tensor] = []
        tensor_count = int(self.subgraph.TensorsLength())
        for index in range(tensor_count):
            tensor = self.subgraph.Tensors(index)
            if tensor is None:
                raise CompileError(f"TFLite tensor {index} is missing")
            if bool(tensor.IsVariable()):
                raise CompileError(f"TFLite tensor {index} is variable")
            if tensor.Sparsity() is not None:
                raise CompileError(f"TFLite tensor {index} uses unsupported sparse storage")
            if int(tensor.VariantTensorsLength()) != 0:
                raise CompileError(f"TFLite tensor {index} uses variant tensor metadata")
            shape = _vector(tensor.ShapeAsNumpy())
            if any(dimension <= 0 for dimension in shape):
                raise CompileError(f"TFLite tensor {index} has non-static shape {shape}")
            signature = _vector(tensor.ShapeSignatureAsNumpy())
            if signature and (any(dimension <= 0 for dimension in signature) or signature != shape):
                raise CompileError(
                    f"TFLite tensor {index} has dynamic/incompatible shape signature {signature}"
                )
            base_name = _decode_text(tensor.Name(), f"tensor_{index}")
            unique_name = base_name if base_name not in names else f"{base_name}__{index}"
            names.add(unique_name)
            quantization = tensor.Quantization()
            if quantization is None:
                scales: tuple[float, ...] = ()
                zero_points: tuple[int, ...] = ()
                quantized_dimension = 0
            else:
                if int(quantization.DetailsType()) != 0:
                    raise CompileError(
                        f"TFLite tensor {index} uses unsupported custom quantization details"
                    )
                scales = _float_vector(quantization.ScaleAsNumpy())
                zero_points = _vector(quantization.ZeroPointAsNumpy())
                quantized_dimension = int(quantization.QuantizedDimension())
            buffer_index = int(tensor.Buffer())
            if not 0 <= buffer_index < int(self.model.BuffersLength()):
                raise CompileError(f"TFLite tensor {index} has invalid buffer index {buffer_index}")
            buffer = self.model.Buffers(buffer_index)
            if buffer is None:
                raise CompileError(f"TFLite tensor {index} references a missing buffer")
            if int(buffer.Offset()) != 0 or int(buffer.Size()) != 0:
                raise CompileError(
                    f"TFLite tensor {index} uses an unsupported external buffer"
                )
            if int(buffer.DataLength()):
                data_value = buffer.DataAsNumpy()
                data = np.asarray(data_value, dtype=np.uint8).tobytes()
            else:
                data = None
            records.append(
                _Tensor(
                    index=index,
                    name=unique_name,
                    shape=shape,
                    type_code=int(tensor.Type()),
                    buffer_index=buffer_index,
                    scales=scales,
                    zero_points=zero_points,
                    quantized_dimension=quantized_dimension,
                    data=data,
                )
            )
        return tuple(records)

    def _read_opcodes(self) -> tuple[_OperatorCode, ...]:
        result: list[_OperatorCode] = []
        for index in range(int(self.model.OperatorCodesLength())):
            code = self.model.OperatorCodes(index)
            if code is None:
                raise CompileError(f"TFLite operator code {index} is missing")
            custom = code.CustomCode()
            result.append(
                _OperatorCode(
                    builtin=int(code.BuiltinCode()),
                    version=int(code.Version()),
                    custom_code=None if custom is None else _decode_text(custom, "<custom>"),
                )
            )
        return tuple(result)

    def _tensor(self, index: int, description: str) -> _Tensor:
        if not 0 <= index < len(self.tensors):
            raise CompileError(f"{description} references invalid tensor index {index}")
        return self.tensors[index]

    def _claim(self, index: int, role: str) -> _Tensor:
        tensor = self._tensor(index, role)
        previous = self.roles.get(index)
        if previous is not None and previous != role:
            raise CompileError(
                f"TFLite tensor {index} is used as incompatible roles {previous} and {role}"
            )
        self.roles[index] = role
        return tensor

    def _decode_constant(self, tensor: _Tensor, dtype: np.dtype[Any]) -> np.ndarray:
        if tensor.data is None:
            raise CompileError(f"TFLite constant tensor {tensor.index} has no embedded data")
        element_count = math.prod(tensor.shape) if tensor.shape else 1
        expected_bytes = element_count * dtype.itemsize
        if len(tensor.data) != expected_bytes:
            raise CompileError(
                f"TFLite constant tensor {tensor.index} has {len(tensor.data)} bytes, "
                f"expected {expected_bytes}"
            )
        return np.frombuffer(tensor.data, dtype=dtype).reshape(tensor.shape).copy()

    def _per_tensor(self, tensor: _Tensor, description: str) -> PerTensorQParams:
        if len(tensor.scales) != 1 or len(tensor.zero_points) != 1:
            raise CompileError(f"{description} requires exactly one scale and zero point")
        try:
            return PerTensorQParams(tensor.scales[0], tensor.zero_points[0])
        except ValueError as error:
            raise CompileError(f"{description} has invalid quantization: {error}") from error

    def _activation(self, index: int, *, output: bool = False) -> str:
        tensor = self._claim(index, "activation")
        if tensor.type_code != int(self.schema.TensorType.INT8):
            raise CompileError(f"TFLite activation {tensor.name} must be INT8")
        if len(tensor.shape) not in (2, 3, 4) or tensor.shape[0] != 1:
            raise CompileError(
                f"TFLite activation {tensor.name} must be static batch-one rank 2, 3, or 4"
            )
        if output and tensor.data is not None:
            raise CompileError(f"TFLite operation output {tensor.name} cannot be a constant")
        if index in self.input_indices and tensor.data is not None:
            raise CompileError(f"TFLite graph input {tensor.name} cannot be a constant")
        layout = Layout.NC if len(tensor.shape) == 2 else Layout.NLC if len(tensor.shape) == 3 else Layout.NHWC
        tensor_type = TensorType(
            tensor.shape,
            DType.INT8,
            layout,
            self._per_tensor(tensor, f"TFLite activation {tensor.name}"),
        )
        previous = self.values.get(tensor.name)
        if previous is not None and previous != tensor_type:
            raise CompileError(f"TFLite tensor {tensor.name} has inconsistent activation types")
        self.values[tensor.name] = tensor_type
        if tensor.data is not None and tensor.name not in self.constants:
            self.constants[tensor.name] = self._decode_constant(tensor, np.dtype("i1"))
        if output:
            if index in self.produced or index in self.input_indices:
                raise CompileError(f"TFLite tensor {tensor.name} has multiple producers")
            self.produced.add(index)
        return tensor.name

    def _weight_qparams(
        self, tensor: _Tensor, *, source_axis: int, target_axis: int, channels: int
    ) -> PerAxisQParams:
        if len(tensor.scales) == 1 and len(tensor.zero_points) == 1:
            scales = tensor.scales * channels
            zero_points = tensor.zero_points * channels
        elif len(tensor.scales) == channels and len(tensor.zero_points) == channels:
            if tensor.quantized_dimension != source_axis:
                raise CompileError(
                    f"TFLite weight {tensor.name} quantized_dimension must be {source_axis}"
                )
            scales = tensor.scales
            zero_points = tensor.zero_points
        else:
            raise CompileError(
                f"TFLite weight {tensor.name} requires one or {channels} qparams"
            )
        if any(zero_point != 0 for zero_point in zero_points):
            raise CompileError(f"TFLite weight {tensor.name} must have zero point 0")
        try:
            return PerAxisQParams(scales, zero_points, target_axis)
        except ValueError as error:
            raise CompileError(f"TFLite weight {tensor.name} has invalid qparams: {error}") from error

    def _weight(
        self, index: int, *, kind: str
    ) -> tuple[str, PerAxisQParams, np.ndarray, int]:
        tensor = self._claim(index, kind)
        if tensor.type_code != int(self.schema.TensorType.INT8):
            raise CompileError(f"TFLite weight {tensor.name} must be INT8")
        stored = self._decode_constant(tensor, np.dtype("i1"))
        if np.any(stored == -128):
            raise CompileError(f"TFLite symmetric weight {tensor.name} contains -128")
        if kind == "conv_weight":
            if len(tensor.shape) != 4:
                raise CompileError(f"TFLite Conv2D weight {tensor.name} must be OHWI rank four")
            channels = tensor.shape[0]
            layout = Layout.OHWI
            qparams = self._weight_qparams(
                tensor, source_axis=0, target_axis=0, channels=channels
            )
        elif kind == "depthwise_weight":
            if len(tensor.shape) != 4 or tensor.shape[0] != 1:
                raise CompileError(
                    f"TFLite depthwise weight {tensor.name} must have shape [1,H,W,O]"
                )
            channels = tensor.shape[3]
            layout = Layout.HWO
            qparams = self._weight_qparams(
                tensor, source_axis=3, target_axis=2, channels=channels
            )
            stored = stored[0].copy()
        elif kind == "linear_weight":
            if len(tensor.shape) != 2:
                raise CompileError(f"TFLite FullyConnected weight {tensor.name} must be OI rank two")
            channels = tensor.shape[0]
            layout = Layout.OI
            qparams = self._weight_qparams(
                tensor, source_axis=0, target_axis=0, channels=channels
            )
        else:  # pragma: no cover - internal dispatch guard
            raise AssertionError(kind)
        tensor_type = TensorType(tuple(stored.shape), DType.INT8, layout, qparams)
        self.values[tensor.name] = tensor_type
        self.constants[tensor.name] = stored
        return tensor.name, qparams, stored, channels

    def _bias(
        self,
        index: int | None,
        *,
        op_name: str,
        input_qparams: PerTensorQParams,
        weight_qparams: PerAxisQParams,
        channels: int,
    ) -> str:
        try:
            expected_scales = tuple(
                normalize_scale_float32(input_qparams.scale * scale)
                for scale in weight_qparams.scales
            )
        except ValueError as error:
            raise CompileError(
                f"{op_name}: bias scale product is not representable as float32"
            ) from error
        if index is None:
            name = f"{op_name}.zero_bias"
            suffix = 0
            while name in self.values:
                suffix += 1
                name = f"{op_name}.zero_bias_{suffix}"
            qparams = PerAxisQParams(expected_scales, (0,) * channels, 0)
            self.values[name] = TensorType((channels,), DType.INT32, Layout.C, qparams)
            self.constants[name] = np.zeros((channels,), dtype=np.int32)
            return name

        tensor = self._claim(index, "bias")
        if tensor.type_code != int(self.schema.TensorType.INT32) or tensor.shape != (channels,):
            raise CompileError(
                f"TFLite bias {tensor.name} must be INT32 with shape ({channels},)"
            )
        if len(tensor.scales) == 1 and len(tensor.zero_points) == 1:
            actual_scales = tensor.scales * channels
            actual_zero_points = tensor.zero_points * channels
        elif len(tensor.scales) == channels and len(tensor.zero_points) == channels:
            if tensor.quantized_dimension != 0:
                raise CompileError(f"TFLite bias {tensor.name} quantized_dimension must be 0")
            actual_scales = tensor.scales
            actual_zero_points = tensor.zero_points
        else:
            raise CompileError(f"TFLite bias {tensor.name} has incompatible qparam count")
        if any(zero_point != 0 for zero_point in actual_zero_points):
            raise CompileError(f"TFLite bias {tensor.name} must have zero point 0")
        normalized_actual = tuple(normalize_scale_float32(scale) for scale in actual_scales)
        if normalized_actual != expected_scales:
            raise CompileError(
                f"TFLite bias {tensor.name} scale must equal input_scale * weight_scale[channel]"
            )
        values = self._decode_constant(tensor, np.dtype("<i4")).astype(np.int32, copy=False)
        qparams = PerAxisQParams(expected_scales, (0,) * channels, 0)
        self.values[tensor.name] = TensorType((channels,), DType.INT32, Layout.C, qparams)
        self.constants[tensor.name] = values
        return tensor.name

    def _constant_ints(self, index: int, description: str) -> tuple[int, ...]:
        tensor = self._tensor(index, description)
        if tensor.type_code == int(self.schema.TensorType.INT32):
            dtype = np.dtype("<i4")
        elif tensor.type_code == int(self.schema.TensorType.INT64):
            dtype = np.dtype("<i8")
        else:
            raise CompileError(f"{description} must be an INT32 or INT64 constant")
        return tuple(int(value) for value in self._decode_constant(tensor, dtype).reshape(-1))

    def _activation_bounds(self, activation: int, output_name: str) -> tuple[int, int]:
        if activation == int(self.schema.ActivationFunctionType.NONE):
            return -128, 127
        qparams = self.values[output_name].qparams
        if not isinstance(qparams, PerTensorQParams):  # pragma: no cover - activation invariant
            raise CompileError("TFLite fused activation requires per-tensor output qparams")
        zero = round_half_away_from_zero(0.0 / qparams.scale) + qparams.zero_point
        zero = max(-128, min(127, zero))
        if activation == int(self.schema.ActivationFunctionType.RELU):
            return zero, 127
        if activation == int(self.schema.ActivationFunctionType.RELU6):
            six = round_half_away_from_zero(6.0 / qparams.scale) + qparams.zero_point
            return zero, max(-128, min(127, six))
        raise CompileError(
            f"unsupported TFLite fused activation value {activation}; only NONE, RELU and RELU6 are supported"
        )

    def _padding(
        self,
        padding: int,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        kernel: tuple[int, int],
        stride: tuple[int, int],
        dilation: tuple[int, int] = (1, 1),
    ) -> tuple[int, int, int, int]:
        if padding == int(self.schema.Padding.VALID):
            return (0, 0, 0, 0)
        if padding != int(self.schema.Padding.SAME):
            raise CompileError(f"unsupported TFLite padding enum {padding}")
        expected_output = (
            math.ceil(input_shape[1] / stride[0]),
            math.ceil(input_shape[2] / stride[1]),
        )
        if output_shape[1:3] != expected_output:
            raise CompileError(
                f"TFLite SAME output {output_shape[1:3]} does not match {expected_output}"
            )
        effective_h = (kernel[0] - 1) * dilation[0] + 1
        effective_w = (kernel[1] - 1) * dilation[1] + 1
        total_h = max(0, (output_shape[1] - 1) * stride[0] + effective_h - input_shape[1])
        total_w = max(0, (output_shape[2] - 1) * stride[1] + effective_w - input_shape[2])
        return (
            total_h // 2,
            total_h - total_h // 2,
            total_w // 2,
            total_w - total_w // 2,
        )

    def _operator_io(self, operator: object, op_name: str) -> tuple[tuple[int, ...], int]:
        inputs = _vector(operator.InputsAsNumpy())
        outputs = _vector(operator.OutputsAsNumpy())
        if len(outputs) != 1 or outputs[0] < 0:
            raise CompileError(f"{op_name}: exactly one non-optional output is required")
        if int(operator.IntermediatesLength()) != 0:
            raise CompileError(f"{op_name}: explicit intermediate tensors are unsupported")
        mutating = _vector(operator.MutatingVariableInputsAsNumpy())
        if any(mutating):
            raise CompileError(f"{op_name}: mutating variable inputs are unsupported")
        if int(operator.CustomOptionsLength()) != 0:
            raise CompileError(f"{op_name}: custom options are unsupported")
        if int(operator.CustomOptionsFormat()) != 0:
            raise CompileError(f"{op_name}: custom options format is unsupported")
        if int(operator.LargeCustomOptionsOffset()) != 0 or int(operator.LargeCustomOptionsSize()) != 0:
            raise CompileError(f"{op_name}: external custom options are unsupported")
        if hasattr(operator, "BuiltinOptions2Type") and int(operator.BuiltinOptions2Type()) != 0:
            raise CompileError(f"{op_name}: secondary builtin options are unsupported")
        return inputs, outputs[0]

    def _conv(self, operator: object, op_name: str, *, depthwise: bool) -> None:
        inputs, output_index = self._operator_io(operator, op_name)
        if len(inputs) not in (2, 3) or any(index < 0 for index in inputs[:2]):
            raise CompileError(f"{op_name}: convolution requires input, weight and optional bias")
        input_name = self._activation(inputs[0])
        output_name = self._activation(output_index, output=True)
        input_type = self.values[input_name]
        output_type = self.values[output_name]
        if len(input_type.shape) != 4 or len(output_type.shape) != 4:
            raise CompileError(f"{op_name}: convolution activations must be rank-four NHWC")
        if depthwise:
            options = _option(
                self.schema,
                operator,
                "DepthwiseConv2DOptions",
                int(self.schema.BuiltinOptions.DepthwiseConv2DOptions),
            )
            weight_name, weight_qparams, weight_values, channels = self._weight(
                inputs[1], kind="depthwise_weight"
            )
            kernel = tuple(int(value) for value in weight_values.shape[:2])
            depth_multiplier = int(options.DepthMultiplier())
        else:
            options = _option(
                self.schema,
                operator,
                "Conv2DOptions",
                int(self.schema.BuiltinOptions.Conv2DOptions),
            )
            weight_name, weight_qparams, weight_values, channels = self._weight(
                inputs[1], kind="conv_weight"
            )
            kernel = tuple(int(value) for value in weight_values.shape[1:3])
            depth_multiplier = 1
            if weight_values.shape[3] != input_type.shape[3]:
                raise CompileError(f"{op_name}: grouped TFLite Conv2D is not supported")
        stride = (int(options.StrideH()), int(options.StrideW()))
        dilation = (int(options.DilationHFactor()), int(options.DilationWFactor()))
        if hasattr(options, "QuantizedBiasType") and int(options.QuantizedBiasType()) != 0:
            raise CompileError(f"{op_name}: explicit quantized_bias_type is unsupported")
        if min(*stride, *dilation) <= 0:
            raise CompileError(f"{op_name}: stride and dilation must be positive")
        padding = self._padding(
            int(options.Padding()), input_type.shape, output_type.shape, kernel, stride, dilation
        )
        bias_index = inputs[2] if len(inputs) == 3 and inputs[2] >= 0 else None
        input_qparams = input_type.qparams
        assert isinstance(input_qparams, PerTensorQParams)
        bias_name = self._bias(
            bias_index,
            op_name=op_name,
            input_qparams=input_qparams,
            weight_qparams=weight_qparams,
            channels=channels,
        )
        activation_min, activation_max = self._activation_bounds(
            int(options.FusedActivationFunction()), output_name
        )
        if depthwise:
            self.ops.append(
                DepthwiseConv2DOp(
                    op_name,
                    input_name,
                    weight_name,
                    bias_name,
                    output_name,
                    depth_multiplier=depth_multiplier,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    activation_min=activation_min,
                    activation_max=activation_max,
                )
            )
        else:
            self.ops.append(
                Conv2DOp(
                    op_name,
                    input_name,
                    weight_name,
                    bias_name,
                    output_name,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    groups=1,
                    activation_min=activation_min,
                    activation_max=activation_max,
                )
            )

    def _fully_connected(self, operator: object, op_name: str, *, version: int) -> None:
        inputs, output_index = self._operator_io(operator, op_name)
        if len(inputs) not in (2, 3) or any(index < 0 for index in inputs[:2]):
            raise CompileError(f"{op_name}: FullyConnected requires input, weight and optional bias")
        options = _option(
            self.schema,
            operator,
            "FullyConnectedOptions",
            int(self.schema.BuiltinOptions.FullyConnectedOptions),
        )
        if int(options.WeightsFormat()) != int(self.schema.FullyConnectedOptionsWeightsFormat.DEFAULT):
            raise CompileError(f"{op_name}: non-default FullyConnected weight format is unsupported")
        if bool(options.KeepNumDims()) or bool(options.AsymmetricQuantizeInputs()):
            raise CompileError(
                f"{op_name}: keep_num_dims and asymmetric_quantize_inputs must be false"
            )
        if int(options.QuantizedBiasType()) != 0:
            raise CompileError(f"{op_name}: explicit quantized_bias_type is unsupported")
        bias_is_present = len(inputs) == 3 and inputs[2] >= 0
        expected_version = 4 if bias_is_present else 6
        if version != expected_version:
            raise CompileError(
                f"{op_name}: FullyConnected with "
                f"{'a bias' if bias_is_present else 'no bias'} requires version "
                f"{expected_version}, got {version}"
            )
        input_name = self._activation(inputs[0])
        output_name = self._activation(output_index, output=True)
        if len(self.values[input_name].shape) != 2 or len(self.values[output_name].shape) != 2:
            raise CompileError(f"{op_name}: FullyConnected activations must be rank-two NC")
        weight_name, weight_qparams, _, channels = self._weight(
            inputs[1], kind="linear_weight"
        )
        input_qparams = self.values[input_name].qparams
        assert isinstance(input_qparams, PerTensorQParams)
        bias_index = inputs[2] if len(inputs) == 3 and inputs[2] >= 0 else None
        bias_name = self._bias(
            bias_index,
            op_name=op_name,
            input_qparams=input_qparams,
            weight_qparams=weight_qparams,
            channels=channels,
        )
        activation_min, activation_max = self._activation_bounds(
            int(options.FusedActivationFunction()), output_name
        )
        self.ops.append(
            LinearOp(
                op_name,
                input_name,
                weight_name,
                bias_name,
                output_name,
                activation_min=activation_min,
                activation_max=activation_max,
            )
        )

    def _add(self, operator: object, op_name: str) -> None:
        inputs, output_index = self._operator_io(operator, op_name)
        if len(inputs) != 2 or any(index < 0 for index in inputs):
            raise CompileError(f"{op_name}: Add requires two inputs")
        options = _option(
            self.schema,
            operator,
            "AddOptions",
            int(self.schema.BuiltinOptions.AddOptions),
        )
        input_a = self._activation(inputs[0])
        input_b = self._activation(inputs[1])
        output = self._activation(output_index, output=True)
        activation_min, activation_max = self._activation_bounds(
            int(options.FusedActivationFunction()), output
        )
        self.ops.append(
            AddOp(
                op_name,
                input_a,
                input_b,
                output,
                activation_min=activation_min,
                activation_max=activation_max,
            )
        )

    def _pool(self, operator: object, op_name: str, *, average: bool) -> None:
        inputs, output_index = self._operator_io(operator, op_name)
        if len(inputs) != 1 or inputs[0] < 0:
            raise CompileError(f"{op_name}: pooling requires one input")
        options = _option(
            self.schema,
            operator,
            "Pool2DOptions",
            int(self.schema.BuiltinOptions.Pool2DOptions),
        )
        input_name = self._activation(inputs[0])
        output_name = self._activation(output_index, output=True)
        input_type = self.values[input_name]
        output_type = self.values[output_name]
        if len(input_type.shape) != 4 or len(output_type.shape) != 4:
            raise CompileError(f"{op_name}: pooling activations must be rank-four NHWC")
        kernel = (int(options.FilterHeight()), int(options.FilterWidth()))
        stride = (int(options.StrideH()), int(options.StrideW()))
        if min(*kernel, *stride) <= 0:
            raise CompileError(f"{op_name}: pool kernel and stride must be positive")
        padding = self._padding(
            int(options.Padding()), input_type.shape, output_type.shape, kernel, stride
        )
        activation_min, activation_max = self._activation_bounds(
            int(options.FusedActivationFunction()), output_name
        )
        if average:
            self.ops.append(
                AveragePool2DOp(
                    op_name,
                    input_name,
                    output_name,
                    kernel=kernel,
                    stride=stride,
                    padding=padding,
                    activation_min=activation_min,
                    activation_max=activation_max,
                    arithmetic_profile=AVERAGE_POOL_PROFILE_TFLITE_RAW_V1,
                )
            )
        else:
            self.ops.append(
                MaxPool2DOp(
                    op_name,
                    input_name,
                    output_name,
                    kernel=kernel,
                    stride=stride,
                    padding=padding,
                    activation_min=activation_min,
                    activation_max=activation_max,
                )
            )

    def _reshape(self, operator: object, op_name: str) -> None:
        inputs, output_index = self._operator_io(operator, op_name)
        if len(inputs) not in (1, 2) or inputs[0] < 0:
            raise CompileError(f"{op_name}: Reshape requires data and an optional shape constant")
        options = _option(
            self.schema,
            operator,
            "ReshapeOptions",
            int(self.schema.BuiltinOptions.ReshapeOptions),
        )
        input_name = self._activation(inputs[0])
        output_name = self._activation(output_index, output=True)
        input_type = self.values[input_name]
        output_type = self.values[output_name]
        if len(inputs) == 2 and inputs[1] >= 0:
            shape_spec = self._constant_ints(inputs[1], f"{op_name} shape")
        else:
            shape_spec = _vector(options.NewShapeAsNumpy())
        _normalize_shape_spec(shape_spec, input_type.shape, output_type.shape)
        flatten_shape = (1, math.prod(input_type.shape[1:]))
        if input_type.layout in (Layout.NHWC, Layout.NLC) and output_type.shape == flatten_shape:
            self.ops.append(FlattenOp(op_name, input_name, output_name))
        else:
            self.ops.append(ReshapeOp(op_name, input_name, output_name))

    def _pad(self, operator: object, op_name: str, *, explicit_value: bool) -> None:
        inputs, output_index = self._operator_io(operator, op_name)
        expected_count = 3 if explicit_value else 2
        if len(inputs) != expected_count or any(index < 0 for index in inputs):
            raise CompileError(f"{op_name}: Pad input count is invalid")
        option_name = "PadV2Options" if explicit_value else "PadOptions"
        option_type = (
            int(self.schema.BuiltinOptions.PadV2Options)
            if explicit_value
            else int(self.schema.BuiltinOptions.PadOptions)
        )
        _option(self.schema, operator, option_name, option_type)
        input_name = self._activation(inputs[0])
        output_name = self._activation(output_index, output=True)
        input_type = self.values[input_name]
        output_type = self.values[output_name]
        if len(input_type.shape) != 4 or len(output_type.shape) != 4:
            raise CompileError(f"{op_name}: BakeNN supports only rank-four NHWC Pad")
        paddings_tensor = self._tensor(inputs[1], f"{op_name} paddings")
        paddings_values = self._constant_ints(inputs[1], f"{op_name} paddings")
        if paddings_tensor.shape != (4, 2) or len(paddings_values) != 8:
            raise CompileError(f"{op_name}: paddings must have shape [4,2]")
        paddings = np.asarray(paddings_values, dtype=np.int64).reshape(4, 2)
        if np.any(paddings < 0) or tuple(paddings[0]) != (0, 0) or tuple(paddings[3]) != (0, 0):
            raise CompileError(f"{op_name}: batch/channel padding is unsupported")
        input_qparams = input_type.qparams
        output_qparams = output_type.qparams
        if input_qparams != output_qparams:
            raise CompileError(f"{op_name}: Pad must preserve input/output qparams")
        assert isinstance(input_qparams, PerTensorQParams)
        if explicit_value:
            pad_tensor = self._tensor(inputs[2], f"{op_name} padding value")
            if pad_tensor.type_code != int(self.schema.TensorType.INT8):
                raise CompileError(f"{op_name}: padding value must be INT8")
            pad_values = self._decode_constant(pad_tensor, np.dtype("i1")).reshape(-1)
            if pad_values.size != 1 or int(pad_values[0]) != input_qparams.zero_point:
                raise CompileError(
                    f"{op_name}: PADV2 value must equal input zero point {input_qparams.zero_point}"
                )
            if pad_tensor.scales or pad_tensor.zero_points:
                if self._per_tensor(pad_tensor, f"{op_name} padding value") != input_qparams:
                    raise CompileError(f"{op_name}: padding-value qparams must match input")
        self.ops.append(
            Pad2DOp(
                op_name,
                input_name,
                output_name,
                padding=(
                    int(paddings[1, 0]),
                    int(paddings[1, 1]),
                    int(paddings[2, 0]),
                    int(paddings[2, 1]),
                ),
            )
        )

    def build(self) -> QuantizedGraph:
        builtin = self.schema.BuiltinOperator
        supported: dict[int, tuple[str, frozenset[int]]] = {
            int(builtin.CONV_2D): ("CONV_2D", frozenset((3,))),
            int(builtin.DEPTHWISE_CONV_2D): ("DEPTHWISE_CONV_2D", frozenset((3,))),
            int(builtin.FULLY_CONNECTED): ("FULLY_CONNECTED", frozenset((4, 6))),
            int(builtin.ADD): ("ADD", frozenset((2,))),
            int(builtin.AVERAGE_POOL_2D): ("AVERAGE_POOL_2D", frozenset((2,))),
            int(builtin.MAX_POOL_2D): ("MAX_POOL_2D", frozenset((2,))),
            int(builtin.RESHAPE): ("RESHAPE", frozenset((1,))),
            int(builtin.PAD): ("PAD", frozenset((2,))),
            int(builtin.PADV2): ("PADV2", frozenset((2,))),
        }
        for op_index in range(int(self.subgraph.OperatorsLength())):
            operator = self.subgraph.Operators(op_index)
            if operator is None:
                raise CompileError(f"TFLite operator {op_index} is missing")
            opcode_index = int(operator.OpcodeIndex())
            if not 0 <= opcode_index < len(self.opcodes):
                raise CompileError(f"TFLite operator {op_index} has invalid opcode index")
            code = self.opcodes[opcode_index]
            if code.builtin == int(builtin.CUSTOM) or code.custom_code is not None:
                raise CompileError(
                    Diagnostic(
                        code="BAKENN_TFLITE_CUSTOM_OPERATOR_UNSUPPORTED",
                        stage="tflite_import",
                        location=f"subgraph[0]/operator[{op_index}]",
                        reason=(
                            "unsupported custom op "
                            f"{code.custom_code or '<unnamed>'}"
                        ),
                        suggestions=(
                            "Convert the custom operation to a supported TFLite builtin",
                        ),
                    )
                )
            specification = supported.get(code.builtin)
            if specification is None:
                raise CompileError(
                    Diagnostic(
                        code="BAKENN_TFLITE_OPERATOR_UNSUPPORTED",
                        stage="tflite_import",
                        location=f"subgraph[0]/operator[{op_index}]",
                        reason=f"unsupported TFLite builtin operator code {code.builtin}",
                        suggestions=(
                            "Use one of the documented static INT8 importer operators",
                            "Open an operator request with the model and operator version",
                        ),
                    )
                )
            kind, versions = specification
            if code.version not in versions:
                allowed = ", ".join(str(version) for version in sorted(versions))
                raise CompileError(
                    Diagnostic(
                        code="BAKENN_TFLITE_OPERATOR_VERSION_UNSUPPORTED",
                        stage="tflite_import",
                        location=f"subgraph[0]/operator[{op_index}]",
                        reason=(
                            f"TFLite {kind} version {code.version} is unsupported; "
                            f"allowed: {allowed}"
                        ),
                        suggestions=(
                            "Re-export the model with a documented operator version",
                        ),
                    )
                )
            op_name = f"tflite_{op_index}_{kind.lower()}"
            if kind == "CONV_2D":
                self._conv(operator, op_name, depthwise=False)
            elif kind == "DEPTHWISE_CONV_2D":
                self._conv(operator, op_name, depthwise=True)
            elif kind == "FULLY_CONNECTED":
                self._fully_connected(operator, op_name, version=code.version)
            elif kind == "ADD":
                self._add(operator, op_name)
            elif kind == "AVERAGE_POOL_2D":
                self._pool(operator, op_name, average=True)
            elif kind == "MAX_POOL_2D":
                self._pool(operator, op_name, average=False)
            elif kind == "RESHAPE":
                self._reshape(operator, op_name)
            elif kind == "PAD":
                self._pad(operator, op_name, explicit_value=False)
            elif kind == "PADV2":
                self._pad(operator, op_name, explicit_value=True)
            else:  # pragma: no cover - exhaustive mapping above
                raise AssertionError(kind)

        input_name = self._activation(self.input_indices[0])
        output_name = self._activation(self.output_indices[0])
        graph = QuantizedGraph(
            name=self.graph_name,
            values=self.values,
            constants=self.constants,
            ops=tuple(self.ops),
            inputs=(input_name,),
            outputs=(output_name,),
        )
        try:
            verify_graph(graph)
        except BakeNNError as error:
            raise CompileError(f"TFLite model is incompatible with BakeNN IR: {error}") from error
        return graph


def import_tflite(source: object, name: str | None = None) -> QuantizedGraph:
    """Import one static fully-quantized TFLite model into BakeNN IR.

    ``source`` may be a filesystem path or bytes-like object.  Only the declared
    fail-closed INT8 subset is accepted; there is no float fallback or retained
    TFLite runtime dependency.
    """

    if name is not None and (not isinstance(name, str) or not name):
        raise CompileError("TFLite graph name must be a non-empty string when provided")
    schema = _optional_dependencies()
    data, source_stem = _read_source(source)
    if len(data) < 8 or data[4:8] != _FILE_IDENTIFIER:
        raise CompileError("input is not a TFLite FlatBuffer with TFL3 identifier")
    try:
        model = schema.Model.GetRootAsModel(data, 0)
        if int(model.Version()) != _SCHEMA_VERSION:
            raise CompileError(
                f"unsupported TFLite schema version {int(model.Version())}; expected {_SCHEMA_VERSION}"
            )
        if int(model.SubgraphsLength()) != 1:
            raise CompileError("TFLite import requires exactly one subgraph")
        subgraph = model.Subgraphs(0)
        if subgraph is None:
            raise CompileError("TFLite model is missing subgraph 0")
        inputs = _vector(subgraph.InputsAsNumpy())
        outputs = _vector(subgraph.OutputsAsNumpy())
        if len(inputs) != 1 or len(outputs) != 1 or inputs[0] < 0 or outputs[0] < 0:
            raise CompileError("TFLite import requires exactly one input and one output")
        graph_name = name or _decode_text(subgraph.Name(), source_stem or "tflite_model")
        if not graph_name:
            raise CompileError("BakeNN graph name must be non-empty")
        return _Importer(schema, model, subgraph, graph_name).build()
    except CompileError:
        raise
    except Exception as error:
        raise CompileError(f"malformed or unsupported TFLite FlatBuffer: {error}") from error


__all__ = ["import_tflite"]
