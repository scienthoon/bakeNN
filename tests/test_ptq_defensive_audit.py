"""Use original-framework semantics, rather than a shared lowering, as oracle."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from bakenn.errors import CompileError
from bakenn.frontends.torch_export import capture_torch_export
from bakenn.plan import lower_to_plan
from bakenn.quantization.ptq import _from_torch_sequential, quantize_ptq
from bakenn.quantization.ptq_graph import _evaluate, quantize_float_graph
from bakenn.quantization.verification import verify_ptq_accuracy
from bakenn.reference import dequantize_output, quantize_input, run_reference


@pytest.mark.parametrize("output_size", [(1, 1), (1, 4), (4, 1)])
@pytest.mark.parametrize("align_corners", [True, False])
def test_bilinear_singleton_output_matches_torch(output_size, align_corners) -> None:  # type: ignore[no-untyped-def]
    class Resize(nn.Module):
        def forward(self, value):  # type: ignore[no-untyped-def]
            return torch.nn.functional.interpolate(value, size=output_size, mode="bilinear", align_corners=align_corners)

    sample = torch.arange(1, 19, dtype=torch.float32).reshape(1, 2, 3, 3)
    model = Resize().eval()
    float_graph = capture_torch_export(model, sample)
    expected = model(sample).numpy()
    actual = _evaluate(float_graph, sample.numpy())[float_graph.outputs[0]]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)


def test_legacy_sequential_preserves_repeated_linear_calls() -> None:
    layer = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        layer.weight.fill_(2.0)
    model = nn.Sequential(layer, layer).eval()
    sample = np.array([[1.0]], dtype=np.float32)
    assert len(_from_torch_sequential(model).layers) == 2
    graph = quantize_ptq(model, sample)
    assert len(graph.ops) == 2
    plan = lower_to_plan(graph)
    result = dequantize_output(plan, run_reference(plan, quantize_input(plan, sample)))
    np.testing.assert_allclose(result, model(torch.from_numpy(sample)).detach().numpy(), atol=0.02)


def test_legacy_sequential_preserves_repeated_relu_calls() -> None:
    relu = nn.ReLU()
    first, second = nn.Linear(1, 1), nn.Linear(1, 1)
    model = nn.Sequential(first, relu, second, relu).eval()
    assert all(layer.relu for layer in _from_torch_sequential(model).layers)


@pytest.mark.parametrize("input_shape, output_shape", [((1, 2, 1, 3), (1, 1, 6)), ((1, 2, 3), (1, 1, 1, 6))])
def test_cross_layout_reshape_rejects_channel_reinterpretation(input_shape, output_shape) -> None:  # type: ignore[no-untyped-def]
    class Reshape(nn.Module):
        def forward(self, value):  # type: ignore[no-untyped-def]
            return value.reshape(output_shape)

    sample = torch.arange(1, 7, dtype=torch.float32).reshape(input_shape)
    float_graph = capture_torch_export(Reshape().eval(), sample)
    with pytest.raises(CompileError, match="preserv.*channel"):
        quantize_float_graph(float_graph, sample)


@pytest.mark.parametrize(
    "input_shape, output_shape", [
        ((1, 2, 1, 3), (1, 2, 3)),
        ((1, 2, 3, 1), (1, 2, 3)),
        ((1, 2, 3), (1, 2, 1, 3)),
        ((1, 2, 3), (1, 2, 3, 1)),
    ],
)
def test_cross_layout_singleton_views_preserve_channel_order(input_shape, output_shape) -> None:  # type: ignore[no-untyped-def]
    class Reshape(nn.Module):
        def forward(self, value):  # type: ignore[no-untyped-def]
            return value.reshape(output_shape)

    sample = torch.arange(1, 7, dtype=torch.float32).reshape(input_shape)
    float_graph = capture_torch_export(Reshape().eval(), sample)
    plan = lower_to_plan(quantize_float_graph(float_graph, sample))
    report = verify_ptq_accuracy(float_graph, plan, sample)
    assert report.output.maximum_absolute_error < 0.02


@pytest.mark.parametrize("input_shape", [(1, 2, 1, 3), (1, 2, 3)])
def test_flatten_accuracy_compares_the_same_element_order(input_shape) -> None:  # type: ignore[no-untyped-def]
    class FlattenLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(6, 1, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(1.0)

        def forward(self, value):  # type: ignore[no-untyped-def]
            return self.linear(value.flatten(1))

    sample = torch.arange(1, 7, dtype=torch.float32).reshape(input_shape)
    float_graph = capture_torch_export(FlattenLinear().eval(), sample)
    plan = lower_to_plan(quantize_float_graph(float_graph, sample))
    report = verify_ptq_accuracy(float_graph, plan, sample)
    assert all(layer.maximum_absolute_error < 0.02 for layer in report.layers)
