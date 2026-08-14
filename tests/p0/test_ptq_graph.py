from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import numpy as np
import bakenn

try:
    import pytest
except ImportError:
    pytest = None

if pytest is not None:
    torch = pytest.importorskip("torch")
else:
    try:
        import torch
    except ImportError:
        torch = None

if torch is not None:
    import torch.nn as nn
    import torch.nn.functional as functional

# Install all P0 lowering/reference registrations explicitly; the PTQ module
# itself remains independent from execution backends.
import bakenn.plan.lowering.conv  # noqa: F401
import bakenn.plan.lowering.elementwise  # noqa: F401
import bakenn.plan.lowering.pool  # noqa: F401
import bakenn.plan.lowering.shape  # noqa: F401
import bakenn.plan.lowering.softmax  # noqa: F401
import bakenn.reference.kernels.conv  # noqa: F401
import bakenn.reference.kernels.elementwise  # noqa: F401
import bakenn.reference.kernels.pool  # noqa: F401
import bakenn.reference.kernels.shape  # noqa: F401
import bakenn.reference.kernels.softmax  # noqa: F401
from bakenn.errors import CompileError
from bakenn.frontends.torch_export import capture_torch_export
from bakenn.ir import (
    AddOp,
    ClampOp,
    ConcatenateOp,
    Conv2DOp,
    DepthwiseConv2DOp,
    DType,
    Layout,
    LinearOp,
    MulOp,
    PerTensorQParams,
    RequantizeOp,
    SoftmaxOp,
    verify_graph,
)
from bakenn.plan.lower import lower_to_plan
from bakenn.quantization.ptq_graph import _evaluate, quantize_float_graph
from bakenn.reference.executor import run_reference


def _round_away(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 0.0, np.floor(values + 0.5), np.ceil(values - 0.5))


def _integer_input(graph, source: np.ndarray) -> np.ndarray:  # type: ignore[no-untyped-def]
    tensor = graph.values[graph.inputs[0]]
    qparams = tensor.qparams
    assert isinstance(qparams, PerTensorQParams)
    quantized = _round_away(np.asarray(source, dtype=np.float64) / qparams.scale)
    quantized += qparams.zero_point
    result = np.clip(quantized, -128, 127).astype(np.int8)
    return np.ascontiguousarray(result.transpose(0, 2, 3, 1) if result.ndim == 4 else result)


def _dequantized_output(graph, output: np.ndarray) -> np.ndarray:  # type: ignore[no-untyped-def]
    tensor = graph.values[graph.outputs[0]]
    qparams = tensor.qparams
    assert isinstance(qparams, PerTensorQParams)
    result = (output.astype(np.int32) - qparams.zero_point) * qparams.scale
    return result.transpose(0, 3, 1, 2) if tensor.layout is Layout.NHWC else result


def _assert_deterministic(test: unittest.TestCase, first, second) -> None:  # type: ignore[no-untyped-def]
    test.assertEqual(first.name, second.name)
    test.assertEqual(first.values, second.values)
    test.assertEqual(first.ops, second.ops)
    test.assertEqual(tuple(first.constants), tuple(second.constants))
    for name in first.constants:
        np.testing.assert_array_equal(first.constants[name], second.constants[name])


@unittest.skipIf(torch is None, "PyTorch is not installed")
class GraphPTQTests(unittest.TestCase):
    def test_public_one_call_torch_ptq_compilation(self) -> None:
        model = nn.Sequential(nn.Linear(4, 3), nn.ReLU()).eval()
        example = torch.randn(1, 4)
        calibration = torch.randn(16, 4)
        with tempfile.TemporaryDirectory() as temporary:
            compiled = bakenn.compile_torch_ptq(
                model,
                example,
                calibration,
                Path(temporary),
                name="one_call",
            )
            self.assertEqual(compiled.graph.name, "one_call")
            self.assertEqual(compiled.plan.name, "one_call")
            self.assertTrue(compiled.artifacts.header.exists())
            self.assertTrue(compiled.artifacts.manifest.exists())

    def test_conv_pool_flatten_linear_ptq_is_deterministic_canonical_and_accurate(self) -> None:
        class TinyCNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(1, 2, 3, padding=1, bias=False)
                self.max_pool = nn.MaxPool2d(2)
                self.avg_pool = nn.AvgPool2d(2, count_include_pad=False)
                self.linear = nn.Linear(2 * 2 * 2, 3, bias=False)

            def forward(self, x):  # type: ignore[no-untyped-def]
                x = self.conv(x)
                x = functional.relu(x)
                x = self.max_pool(x)
                x = self.avg_pool(x)
                return self.linear(torch.flatten(x, 1))

        torch.manual_seed(1001)
        model = TinyCNN().eval()
        sample = torch.randn(1, 1, 8, 8)
        calibration = torch.cat((sample, torch.randn(31, 1, 8, 8)), dim=0)
        float_graph = capture_torch_export(model, sample, name="tiny_cnn")
        evaluated = _evaluate(float_graph, sample.numpy())[float_graph.outputs[0]]
        np.testing.assert_allclose(
            evaluated,
            model(sample).detach().numpy(),
            rtol=2.0e-5,
            atol=2.0e-6,
        )

        first = quantize_float_graph(float_graph, calibration)
        second = quantize_float_graph(float_graph, calibration)
        _assert_deterministic(self, first, second)
        verify_graph(first)
        conv = next(op for op in first.ops if isinstance(op, Conv2DOp))
        linear = next(op for op in first.ops if isinstance(op, LinearOp))
        self.assertEqual(conv.output, "relu")
        self.assertFalse(any(isinstance(op, ClampOp) and op.name == "relu" for op in first.ops))
        self.assertFalse(
            any(isinstance(op, RequantizeOp) and op.name.startswith("relu.") for op in first.ops)
        )
        self.assertEqual(first.values[first.inputs[0]].shape, (1, 8, 8, 1))
        self.assertIs(first.values[first.inputs[0]].layout, Layout.NHWC)
        self.assertEqual(first.values[conv.weight].shape, (2, 3, 3, 1))
        self.assertIs(first.values[conv.weight].layout, Layout.OHWI)
        self.assertIs(first.values[linear.weight].layout, Layout.OI)
        np.testing.assert_array_equal(first.constants[conv.bias], np.zeros(2, dtype=np.int32))
        np.testing.assert_array_equal(first.constants[linear.bias], np.zeros(3, dtype=np.int32))

        plan = lower_to_plan(first)
        integer_output = run_reference(plan, _integer_input(first, sample.numpy()))
        dequantized = _dequantized_output(first, integer_output)
        np.testing.assert_allclose(
            dequantized,
            model(sample).detach().numpy(),
            rtol=0.0,
            atol=0.06,
        )

    def test_residual_depthwise_add_relu6_ptq_and_float_evaluator(self) -> None:
        class ResidualDepthwise(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.depthwise = nn.Conv2d(3, 3, 3, padding=1, groups=3)
                self.pool = nn.AvgPool2d(2, count_include_pad=False)

            def forward(self, x):  # type: ignore[no-untyped-def]
                return self.pool(functional.relu6(self.depthwise(x)) + x)

        torch.manual_seed(2002)
        model = ResidualDepthwise().eval()
        sample = torch.randn(1, 3, 6, 6)
        calibration = torch.cat((sample, torch.randn(31, 3, 6, 6)), dim=0)
        float_graph = capture_torch_export(model, sample, name="residual_depthwise")
        np.testing.assert_allclose(
            _evaluate(float_graph, sample.numpy())[float_graph.outputs[0]],
            model(sample).detach().numpy(),
            rtol=3.0e-5,
            atol=3.0e-6,
        )
        graph = quantize_float_graph(float_graph, calibration)
        verify_graph(graph)
        depthwise = next(op for op in graph.ops if isinstance(op, DepthwiseConv2DOp))
        self.assertEqual(depthwise.depth_multiplier, 1)
        self.assertIs(graph.values[depthwise.weight].layout, Layout.HWO)
        self.assertEqual(graph.values[depthwise.weight].shape, (3, 3, 3))
        self.assertTrue(any(isinstance(op, AddOp) for op in graph.ops))

        output = run_reference(lower_to_plan(graph), _integer_input(graph, sample.numpy()))
        np.testing.assert_allclose(
            _dequantized_output(graph, output),
            model(sample).detach().numpy(),
            rtol=0.0,
            atol=0.12,
        )

    def test_concat_mul_softmax_has_fixed_output_qparams_and_hand_float_semantics(self) -> None:
        class CatMulSoftmax(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "factor", torch.linspace(0.5, 1.5, 8, dtype=torch.float32).reshape(1, 8)
                )

            def forward(self, x):  # type: ignore[no-untyped-def]
                return torch.softmax(torch.cat((x, x), dim=1) * self.factor, dim=-1)

        torch.manual_seed(3003)
        model = CatMulSoftmax().eval()
        sample = torch.tensor([[-1.0, 0.25, 1.5, -0.5]], dtype=torch.float32)
        calibration = torch.cat((sample, torch.randn(63, 4)), dim=0)
        float_graph = capture_torch_export(model, sample, name="cat_mul_softmax")
        factor_name = next(
            name for name, value in float_graph.values.items() if value.source_name == "factor"
        )
        logits = np.concatenate((sample.numpy(), sample.numpy()), axis=1)
        logits *= float_graph.constants[factor_name]
        exponentials = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        hand = exponentials / np.sum(exponentials, axis=-1, keepdims=True)
        np.testing.assert_allclose(
            _evaluate(float_graph, sample.numpy())[float_graph.outputs[0]],
            hand,
            rtol=1.0e-6,
            atol=1.0e-7,
        )

        graph = quantize_float_graph(float_graph, calibration)
        verify_graph(graph)
        self.assertTrue(any(isinstance(op, ConcatenateOp) for op in graph.ops))
        self.assertTrue(any(isinstance(op, MulOp) for op in graph.ops))
        self.assertIsInstance(graph.ops[-1], SoftmaxOp)
        output_type = graph.values[graph.outputs[0]]
        self.assertEqual(output_type.qparams, PerTensorQParams(1.0 / 256.0, -128))
        self.assertIs(output_type.dtype, DType.INT8)

        output = run_reference(lower_to_plan(graph), _integer_input(graph, sample.numpy()))
        probabilities = _dequantized_output(graph, output)
        np.testing.assert_allclose(probabilities, hand, rtol=0.0, atol=0.035)

    def test_calibration_and_layout_unsafe_views_fail_closed(self) -> None:
        model = nn.Sequential(nn.Linear(4, 3)).eval()
        float_graph = capture_torch_export(model, torch.randn(1, 4))
        with self.assertRaisesRegex(CompileError, "at least one sample"):
            quantize_float_graph(float_graph, [])
        with self.assertRaisesRegex(CompileError, "incompatible with batch-one input"):
            quantize_float_graph(float_graph, np.zeros((2, 5), dtype=np.float32))
        malformed = np.zeros((2, 4), dtype=np.float32)
        malformed[0, 0] = np.nan
        with self.assertRaisesRegex(CompileError, "NaN or infinity"):
            quantize_float_graph(float_graph, malformed)
        with self.assertRaisesRegex(CompileError, "requires a FloatGraph"):
            quantize_float_graph(object(), np.zeros((1, 4), dtype=np.float32))  # type: ignore[arg-type]

        class UnsafeReshape(nn.Module):
            def forward(self, x):  # type: ignore[no-untyped-def]
                return x.reshape(1, 2, 2, 6)

        unsafe = capture_torch_export(UnsafeReshape().eval(), torch.randn(1, 3, 4, 2))
        with self.assertRaisesRegex(CompileError, "general NCHW Reshape is not layout-preserving"):
            quantize_float_graph(unsafe, np.zeros((1, 3, 4, 2), dtype=np.float32))

        class FlattenOutput(nn.Module):
            def forward(self, x):  # type: ignore[no-untyped-def]
                return torch.flatten(x, 1)

        flatten_output = capture_torch_export(FlattenOutput().eval(), torch.randn(1, 2, 3, 4))
        with self.assertRaisesRegex(CompileError, "supported only when consumed by one Linear"):
            quantize_float_graph(
                flatten_output,
                np.zeros((1, 2, 3, 4), dtype=np.float32),
            )

        constant_channel = nn.Linear(4, 1)
        with torch.no_grad():
            constant_channel.weight.zero_()
            constant_channel.bias.fill_(1.0)
        constant_graph = capture_torch_export(constant_channel.eval(), torch.zeros(1, 4))
        quantized_constant = quantize_float_graph(
            constant_graph, np.zeros((2, 4), dtype=np.float32)
        )
        constant_plan = lower_to_plan(quantized_constant)
        low = run_reference(constant_plan, np.full((1, 4), -128, dtype=np.int8))
        high = run_reference(constant_plan, np.full((1, 4), 127, dtype=np.int8))
        np.testing.assert_array_equal(low, high)
        np.testing.assert_allclose(
            _dequantized_output(quantized_constant, low),
            np.asarray([[1.0]], dtype=np.float32),
            atol=quantized_constant.values[quantized_constant.outputs[0]].qparams.scale,
        )

    def test_calibration_snapshots_reused_buffers_before_iterator_advances(self) -> None:
        float_graph = capture_torch_export(nn.Linear(4, 2, bias=False).eval(), torch.zeros(1, 4))
        backing = np.empty((1, 4), dtype=np.float32)

        def reused_buffer():  # type: ignore[no-untyped-def]
            backing.fill(-1.0)
            yield backing
            backing.fill(10.0)
            yield backing

        streamed = quantize_float_graph(float_graph, reused_buffer())
        copied = quantize_float_graph(
            float_graph,
            (
                np.full((1, 4), -1.0, dtype=np.float32),
                np.full((1, 4), 10.0, dtype=np.float32),
            ),
        )
        self.assertEqual(
            streamed.values[streamed.inputs[0]].qparams,
            copied.values[copied.inputs[0]].qparams,
        )

    def test_single_field_dataloader_batches_and_tuple_batches_are_supported(self) -> None:
        float_graph = capture_torch_export(nn.Linear(4, 2).eval(), torch.zeros(1, 4))
        calibration = torch.linspace(-3.0, 5.0, 32, dtype=torch.float32).reshape(8, 4)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(calibration),
            batch_size=3,
            shuffle=False,
        )
        from_loader = quantize_float_graph(float_graph, loader)

        tuple_batches = ((batch,) for batch in calibration.split(3))
        from_tuples = quantize_float_graph(float_graph, tuple_batches)
        direct = quantize_float_graph(float_graph, calibration)
        self.assertEqual(
            from_loader.values[from_loader.inputs[0]].qparams,
            direct.values[direct.inputs[0]].qparams,
        )
        self.assertEqual(
            from_tuples.values[from_tuples.inputs[0]].qparams,
            direct.values[direct.inputs[0]].qparams,
        )

        ambiguous = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(calibration, torch.arange(8)),
            batch_size=2,
            shuffle=False,
        )
        with self.assertRaisesRegex(CompileError, "multiple fields.*ambiguous"):
            quantize_float_graph(float_graph, ambiguous)

    def test_calibration_rejects_non_real_numeric_dtypes_before_cast(self) -> None:
        float_graph = capture_torch_export(nn.Linear(4, 2).eval(), torch.zeros(1, 4))
        invalid = (
            np.ones((1, 4), dtype=np.complex64) * (1.0 + 2.0j),
            np.asarray([[object(), object(), object(), object()]], dtype=object),
            np.asarray([["1", "2", "3", "4"]], dtype=np.str_),
            np.ones((1, 4), dtype=np.bool_),
        )
        for calibration in invalid:
            with self.subTest(dtype=calibration.dtype):
                with self.assertRaisesRegex(CompileError, "real numeric dtype"):
                    quantize_float_graph(float_graph, calibration)

    def test_graph_activation_zero_point_uses_deployable_float32_scale(self) -> None:
        float_graph = capture_torch_export(nn.Linear(2, 1).eval(), torch.zeros(1, 2))
        calibration = np.asarray(
            [[-1_315_095.0, 133_497.78125]],
            dtype=np.float32,
        )
        graph = quantize_float_graph(float_graph, calibration)
        qparams = graph.values[graph.inputs[0]].qparams
        self.assertEqual(qparams.scale, float(np.float32(5680.756004901961)))
        self.assertEqual(qparams.zero_point, 104)


if __name__ == "__main__":
    unittest.main()
