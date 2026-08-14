from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable

import numpy as np

from bakenn.errors import CompileError
from bakenn.ir import (
    DType,
    Layout,
    LinearOp,
    PerTensorQParams,
    QuantizedGraph,
    TensorType,
    verify_graph,
)
from bakenn.ir.types import normalize_scale_float32
from bakenn.quantization.primitives import quantize_compute_constants
from .fixedpoint import ARITHMETIC_PROFILE


@dataclass(frozen=True)
class FloatLinear:
    weight: np.ndarray
    bias: np.ndarray | None = None
    relu: bool = False
    name: str = "linear"

    def __post_init__(self) -> None:
        raw_weight = np.asarray(self.weight)
        if raw_weight.dtype.kind not in "iuf":
            raise ValueError("FloatLinear weight must contain real numeric values")
        weight = np.asarray(raw_weight, dtype=np.float32)
        if weight.ndim != 2 or 0 in weight.shape or not np.all(np.isfinite(weight)):
            raise ValueError("FloatLinear weight must be a finite [out, in] matrix")
        if self.bias is None:
            bias = np.zeros(weight.shape[0], dtype=np.float32)
        else:
            raw_bias = np.asarray(self.bias)
            if raw_bias.dtype.kind not in "iuf":
                raise ValueError("FloatLinear bias must contain real numeric values")
            bias = np.asarray(raw_bias, dtype=np.float32)
        if bias.shape != (weight.shape[0],) or not np.all(np.isfinite(bias)):
            raise ValueError("FloatLinear bias must have one finite value per output")
        weight = np.array(weight, copy=True, order="C")
        bias = np.array(bias, copy=True, order="C")
        weight.setflags(write=False)
        bias.setflags(write=False)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "bias", bias)


@dataclass(frozen=True)
class FloatMLP:
    layers: tuple[FloatLinear, ...]
    name: str = "model"

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("FloatMLP requires at least one layer")
        for previous, current in zip(self.layers, self.layers[1:]):
            if previous.weight.shape[0] != current.weight.shape[1]:
                raise ValueError("adjacent FloatLinear feature counts do not match")


def _round_array_away(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 0.0, np.floor(values + 0.5), np.ceil(values - 0.5))


def _activation_qparams(values: np.ndarray) -> PerTensorQParams:
    minimum = min(float(np.min(values)), 0.0)
    maximum = max(float(np.max(values)), 0.0)
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise CompileError("calibration data produced non-finite activations")
    if minimum == maximum:
        return PerTensorQParams(scale=1.0, zero_point=0)
    scale = normalize_scale_float32((maximum - minimum) / 255.0)
    zero_point = int(_round_array_away(np.asarray(-128.0 - minimum / scale)).item())
    zero_point = max(-128, min(127, zero_point))
    return PerTensorQParams(scale=scale, zero_point=zero_point)


def _as_numpy_batch(batch: object) -> np.ndarray:
    if isinstance(batch, (tuple, list)):
        if not batch:
            raise CompileError("empty calibration batch")
        if len(batch) != 1:
            raise CompileError(
                "calibration batch sequences must contain exactly one model input"
            )
        batch = batch[0]
    if hasattr(batch, "detach"):
        batch = batch.detach().cpu().numpy()
    raw = np.asarray(batch)
    if raw.dtype.kind not in "iuf":
        raise CompileError("calibration samples must be real numeric values")
    array = np.array(raw, dtype=np.float32, copy=True, order="C")
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise CompileError("v0 calibration samples must have shape [samples, features]")
    return array


def _collect_calibration(calibration_data: object) -> np.ndarray:
    if isinstance(calibration_data, np.ndarray) or hasattr(calibration_data, "detach"):
        batches = [_as_numpy_batch(calibration_data)]
    else:
        batches = [_as_numpy_batch(batch) for batch in calibration_data]  # type: ignore[union-attr]
    if not batches:
        raise CompileError("calibration_data must contain at least one sample")
    result = np.concatenate(batches, axis=0)
    if result.shape[0] == 0:
        raise CompileError("calibration_data must contain at least one sample")
    if not np.all(np.isfinite(result)):
        raise CompileError("calibration_data contains NaN or infinity")
    return result


def _from_torch_sequential(model: object) -> FloatMLP:
    try:
        import torch.nn as nn
    except ImportError as error:
        raise CompileError("PyTorch frontend requested, but torch is not installed") from error
    if not isinstance(model, nn.Sequential):
        raise CompileError("v0 PyTorch frontend supports nn.Sequential only")
    layers: list[FloatLinear] = []
    for module_name, module in model.named_children():
        if isinstance(module, nn.Linear):
            weight = module.weight.detach().cpu().numpy()
            bias = None if module.bias is None else module.bias.detach().cpu().numpy()
            layers.append(FloatLinear(weight=weight, bias=bias, name=module_name))
        elif isinstance(module, nn.ReLU):
            if not layers or layers[-1].relu:
                raise CompileError("ReLU must directly follow one Linear in v0")
            layers[-1] = replace(layers[-1], relu=True)
        else:
            raise CompileError(f"unsupported PyTorch module in v0: {type(module).__name__}")
    return FloatMLP(tuple(layers), name=type(model).__name__.lower())


def quantize_ptq(
    model: FloatMLP | object,
    calibration_data: np.ndarray | Iterable[object],
    *,
    name: str | None = None,
) -> QuantizedGraph:
    """Calibrate and quantize a static MLP into the framework-neutral IR."""
    float_model = model if isinstance(model, FloatMLP) else _from_torch_sequential(model)
    samples = _collect_calibration(calibration_data)
    if samples.shape[1] != float_model.layers[0].weight.shape[1]:
        raise CompileError("calibration feature count does not match model input")

    graph_name = name or float_model.name
    values: dict[str, TensorType] = {}
    constants: dict[str, np.ndarray] = {}
    ops: list[LinearOp] = []

    input_qparams = _activation_qparams(samples)
    current_name = "input"
    current_qparams = input_qparams
    current_activations = samples
    values[current_name] = TensorType(
        shape=(1, samples.shape[1]), dtype=DType.INT8, layout=Layout.NC, qparams=input_qparams
    )

    used_names: set[str] = set()
    for index, layer in enumerate(float_model.layers):
        layer_name = layer.name or f"linear_{index}"
        if layer_name in used_names:
            raise CompileError(f"duplicate layer name: {layer_name}")
        used_names.add(layer_name)

        output_float = current_activations @ layer.weight.T + layer.bias
        if layer.relu:
            output_float = np.maximum(output_float, 0.0)
        output_qparams = _activation_qparams(output_float)
        quantized_weight, quantized_bias = quantize_compute_constants(
            layer.weight,
            layer.bias,
            layout=Layout.OI,
            input_qparams=current_qparams,
            output_qparams=output_qparams,
        )
        weight_q = quantized_weight.values
        weight_qparams = quantized_weight.qparams
        bias_q = quantized_bias.values
        bias_qparams = quantized_bias.qparams

        weight_name = f"{layer_name}.weight"
        bias_name = f"{layer_name}.bias"
        output_name = f"{layer_name}.output"
        values[weight_name] = TensorType(weight_q.shape, DType.INT8, Layout.OI, weight_qparams)
        values[bias_name] = TensorType(bias_q.shape, DType.INT32, Layout.C, bias_qparams)
        values[output_name] = TensorType(
            (1, layer.weight.shape[0]), DType.INT8, Layout.NC, output_qparams
        )
        constants[weight_name] = weight_q
        constants[bias_name] = bias_q
        ops.append(
            LinearOp(
                name=layer_name,
                input=current_name,
                weight=weight_name,
                bias=bias_name,
                output=output_name,
                activation_min=output_qparams.zero_point if layer.relu else -128,
                activation_max=127,
            )
        )
        current_name = output_name
        current_qparams = output_qparams
        current_activations = output_float

    graph = QuantizedGraph(
        name=graph_name,
        values=values,
        constants=constants,
        ops=tuple(ops),
        inputs=("input",),
        outputs=(current_name,),
        arithmetic_profile=ARITHMETIC_PROFILE,
    )
    verify_graph(graph)
    return graph
