from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from bakenn.frontends.torch_export import capture_torch_export
from bakenn.quantization.ptq_graph import quantize_float_graph_with_report
from bakenn.quantization.report import CALIBRATION_REPORT_SCHEMA


def test_ptq_report_records_exact_streaming_calibration_evidence() -> None:
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU()).eval()
    float_graph = capture_torch_export(model, torch.zeros(1, 4), name="reported")
    calibration = (
        np.full((1, 4), value, dtype=np.float32) for value in (-2.0, 0.5, 3.0)
    )

    result = quantize_float_graph_with_report(float_graph, calibration)

    assert result.graph.name == "reported"
    assert result.report.schema == CALIBRATION_REPORT_SCHEMA
    assert result.report.sample_count == 3
    assert result.report.input_shape == (1, 4)
    assert tuple(edge.name for edge in result.report.edges) == tuple(
        name
        for name, value in float_graph.values.items()
        if value.kind.value in {"input", "activation"}
    )
    input_edge = next(
        edge for edge in result.report.edges if edge.name == float_graph.inputs[0]
    )
    assert input_edge.element_count == 12
    assert input_edge.minimum == -2.0
    assert input_edge.maximum == 3.0
    assert input_edge.scale == result.graph.values[result.graph.inputs[0]].qparams.scale
    assert input_edge.zero_point == result.graph.values[result.graph.inputs[0]].qparams.zero_point

    payload = json.loads(result.report.to_json())
    assert payload["sample_count"] == 3
    assert payload["edge_count"] == len(result.report.edges)
    assert result.report.to_json() == result.report.to_json()


def test_ptq_report_calls_out_degenerate_ranges() -> None:
    linear = nn.Linear(4, 2, bias=False).eval()
    with torch.no_grad():
        linear.weight.zero_()
    graph = capture_torch_export(linear, torch.zeros(1, 4), name="degenerate")

    result = quantize_float_graph_with_report(
        graph, np.zeros((2, 4), dtype=np.float32)
    )

    assert result.report.warnings
    assert all("degenerate" in warning for warning in result.report.warnings)
