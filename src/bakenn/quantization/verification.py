"""Layer-wise FP32 versus deployed-INT8 verification diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from bakenn.errors import CompileError
from bakenn.frontends.torch_export.model import FloatGraph, FloatValueKind
from bakenn.ir import DType, PerTensorQParams
from bakenn.plan import ExecutionPlan
from bakenn.quantization.ptq_graph import _canonical_activation, _evaluate, _samples
from bakenn.reference import quantize_input, run_reference_trace


VERIFICATION_REPORT_SCHEMA = "bakenn.ptq-verification.v1"


@dataclass(frozen=True)
class LayerErrorReport:
    name: str
    element_count: int
    maximum_absolute_error: float
    mean_absolute_error: float
    root_mean_square_error: float
    int8_endpoint_count: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("layer report name must be non-empty")
        if self.element_count <= 0:
            raise ValueError("layer report element_count must be positive")
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (
                self.maximum_absolute_error,
                self.mean_absolute_error,
                self.root_mean_square_error,
            )
        ):
            raise ValueError("layer report errors must be finite and non-negative")
        if not 0 <= self.int8_endpoint_count <= self.element_count:
            raise ValueError("layer report endpoint count is invalid")

    @property
    def int8_endpoint_fraction(self) -> float:
        return self.int8_endpoint_count / self.element_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "element_count": self.element_count,
            "maximum_absolute_error": self.maximum_absolute_error,
            "mean_absolute_error": self.mean_absolute_error,
            "root_mean_square_error": self.root_mean_square_error,
            "int8_endpoint_count": self.int8_endpoint_count,
            "int8_endpoint_fraction": self.int8_endpoint_fraction,
        }


@dataclass(frozen=True)
class PTQVerificationReport:
    graph_name: str
    sample_count: int
    layers: tuple[LayerErrorReport, ...]
    schema: str = VERIFICATION_REPORT_SCHEMA

    def __post_init__(self) -> None:
        if not self.graph_name:
            raise ValueError("verification graph_name must be non-empty")
        if self.sample_count <= 0:
            raise ValueError("verification sample_count must be positive")
        if not self.layers:
            raise ValueError("verification report must contain at least one layer")
        if len({layer.name for layer in self.layers}) != len(self.layers):
            raise ValueError("verification layer names must be unique")
        if self.schema != VERIFICATION_REPORT_SCHEMA:
            raise ValueError("unsupported PTQ verification report schema")

    def layer(self, name: str) -> LayerErrorReport:
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise KeyError(name)

    @property
    def output(self) -> LayerErrorReport:
        return self.layers[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "graph_name": self.graph_name,
            "sample_count": self.sample_count,
            "layer_count": len(self.layers),
            "layers": [layer.to_dict() for layer in self.layers],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, indent=2, ensure_ascii=False
        ) + "\n"

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.write_text(self.to_json(), encoding="utf-8")
        return destination


def verify_ptq_accuracy(
    float_graph: FloatGraph,
    plan: ExecutionPlan,
    samples: object,
    *,
    max_samples: int | None = None,
) -> PTQVerificationReport:
    """Measure dequantized deployment error for every common activation edge.

    This is an accuracy diagnostic, not the generated-C correctness oracle.
    Generated C is separately required to be byte-exact with the integer
    reference.  The caller supplies a validation corpus, which may be distinct
    from the representative calibration corpus.
    """

    if max_samples is not None and (
        not isinstance(max_samples, int)
        or isinstance(max_samples, bool)
        or max_samples <= 0
    ):
        raise ValueError("max_samples must be a positive integer or None")
    if not isinstance(float_graph, FloatGraph):
        raise TypeError("float_graph must be a FloatGraph")
    if not isinstance(plan, ExecutionPlan):
        raise TypeError("plan must be an ExecutionPlan")

    comparable_names = tuple(
        name
        for name, value in float_graph.values.items()
        if value.kind in (FloatValueKind.INPUT, FloatValueKind.ACTIVATION)
        and name in plan.tensors
        and plan.tensors[name].tensor_type.dtype is DType.INT8
    )
    if not comparable_names:
        raise CompileError("float graph and execution plan have no common INT8 edges")

    aggregate: dict[str, dict[str, float | int]] = {
        name: {"count": 0, "maximum": 0.0, "absolute_sum": 0.0, "square_sum": 0.0, "endpoints": 0}
        for name in comparable_names
    }
    sample_count = 0
    expected_shape = float_graph.values[float_graph.inputs[0]].shape
    sample_iterator = _samples(samples, expected_shape)
    selected_samples = (
        sample_iterator if max_samples is None else islice(sample_iterator, max_samples)
    )
    for sample in selected_samples:
        sample_count += 1
        float_values = _evaluate(float_graph, sample)
        canonical_input = _canonical_activation(sample)
        trace = run_reference_trace(plan, quantize_input(plan, canonical_input))
        for name in comparable_names:
            if name not in trace:
                # Fused host-only activation names need not survive lowering.
                continue
            tensor_type = plan.tensors[name].tensor_type
            qparams = tensor_type.qparams
            if not isinstance(qparams, PerTensorQParams):
                raise CompileError(f"{name}: activation verification requires per-tensor qparams")
            reference = _canonical_activation(float_values[name]).astype(np.float64)
            quantized = trace[name]
            if reference.shape != quantized.shape:
                raise CompileError(
                    f"{name}: verification shape mismatch {reference.shape} versus {quantized.shape}"
                )
            deployed = (
                quantized.astype(np.int32) - qparams.zero_point
            ).astype(np.float64) * qparams.scale
            difference = np.abs(reference - deployed)
            entry = aggregate[name]
            entry["count"] = int(entry["count"]) + int(difference.size)
            entry["maximum"] = max(float(entry["maximum"]), float(np.max(difference)))
            entry["absolute_sum"] = float(entry["absolute_sum"]) + float(np.sum(difference, dtype=np.float64))
            entry["square_sum"] = float(entry["square_sum"]) + float(
                np.sum(np.square(difference), dtype=np.float64)
            )
            entry["endpoints"] = int(entry["endpoints"]) + int(
                np.count_nonzero((quantized == -128) | (quantized == 127))
            )

    if sample_count == 0:
        raise CompileError("verification samples must contain at least one sample")
    layers: list[LayerErrorReport] = []
    for name in comparable_names:
        entry = aggregate[name]
        count = int(entry["count"])
        if count == 0:
            continue
        layers.append(
            LayerErrorReport(
                name=name,
                element_count=count,
                maximum_absolute_error=float(entry["maximum"]),
                mean_absolute_error=float(entry["absolute_sum"]) / count,
                root_mean_square_error=math.sqrt(float(entry["square_sum"]) / count),
                int8_endpoint_count=int(entry["endpoints"]),
            )
        )
    if not layers:
        raise CompileError("no lowered activation edge was available for verification")
    output_name = plan.outputs[0]
    layers.sort(key=lambda layer: (layer.name == output_name, comparable_names.index(layer.name)))
    return PTQVerificationReport(float_graph.name, sample_count, tuple(layers))


__all__ = [
    "VERIFICATION_REPORT_SCHEMA",
    "LayerErrorReport",
    "PTQVerificationReport",
    "verify_ptq_accuracy",
]
