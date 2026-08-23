from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from bakenn.frontends.torch_export import capture_torch_export
from bakenn.plan import lower_to_plan
from bakenn.quantization.ptq_graph import quantize_float_graph
from bakenn.quantization.verification import verify_ptq_accuracy


def test_layerwise_ptq_verification_reports_deployed_error() -> None:
    torch.manual_seed(91)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2)).eval()
    example = torch.zeros(1, 4)
    float_graph = capture_torch_export(model, example, name="verified")
    calibration = np.linspace(-2.0, 3.0, 48, dtype=np.float32).reshape(12, 4)
    graph = quantize_float_graph(float_graph, calibration)
    plan = lower_to_plan(graph)

    report = verify_ptq_accuracy(float_graph, plan, calibration, max_samples=5)

    assert report.graph_name == "verified"
    assert report.sample_count == 5
    assert report.output.name == plan.outputs[0]
    assert report.output.element_count == 10
    assert report.output.maximum_absolute_error >= 0.0
    assert 0.0 <= report.output.int8_endpoint_fraction <= 1.0
    payload = json.loads(report.to_json())
    assert payload["sample_count"] == 5
    assert payload["layer_count"] == len(report.layers)


def test_layerwise_ptq_verification_validates_sample_limit() -> None:
    with pytest.raises(ValueError, match="max_samples"):
        verify_ptq_accuracy(object(), object(), [], max_samples=0)  # type: ignore[arg-type]


def test_layerwise_ptq_verification_does_not_consume_past_sample_limit() -> None:
    model = nn.Linear(4, 2).eval()
    float_graph = capture_torch_export(model, torch.zeros(1, 4), name="bounded")
    calibration = np.zeros((2, 4), dtype=np.float32)
    plan = lower_to_plan(quantize_float_graph(float_graph, calibration))
    consumed = 0

    def samples():  # type: ignore[no-untyped-def]
        nonlocal consumed
        for value in range(3):
            consumed += 1
            yield np.full((1, 4), value, dtype=np.float32)

    report = verify_ptq_accuracy(float_graph, plan, samples(), max_samples=1)

    assert report.sample_count == 1
    assert consumed == 1
