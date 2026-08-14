from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

try:
    import pytest
except ImportError:  # /usr/local/bin/python3.13 has Torch but intentionally no pytest.
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

from bakenn.errors import CompileError
from bakenn.frontends.torch_export import (
    ALLOWED_ATEN_TARGETS,
    FloatAddOp,
    FloatAveragePool2DOp,
    FloatConcatOp,
    FloatConv2DOp,
    FloatConvTranspose2DOp,
    FloatDepthwiseConv2DOp,
    FloatFlattenOp,
    FloatGraph,
    FloatLayout,
    FloatLinearOp,
    FloatMaxPool2DOp,
    FloatMulOp,
    FloatReLU6Op,
    FloatReLUOp,
    FloatReshapeOp,
    FloatSoftmaxOp,
    FloatSliceOp,
    FloatValueKind,
    capture_torch_export,
)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TorchExportCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        # torch.export shares Dynamo's global compilation cache and recompile
        # budget. Isolate tests without mutating production capture settings.
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()

    def test_real_torch_export_tiny_cnn_extracts_typed_immutable_semantics(self) -> None:
        torch.manual_seed(41)
        model = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(4 * 4 * 4, 5),
            nn.Softmax(dim=-1),
        ).eval()
        graph = capture_torch_export(model, torch.randn(1, 3, 8, 8), name="tiny_cnn")
        self.assertIsInstance(graph, FloatGraph)
        self.assertEqual(graph.name, "tiny_cnn")
        self.assertEqual(
            tuple(type(op) for op in graph.ops),
            (
                FloatConv2DOp,
                FloatReLUOp,
                FloatMaxPool2DOp,
                FloatFlattenOp,
                FloatLinearOp,
                FloatSoftmaxOp,
            ),
        )
        conv = graph.ops[0]
        self.assertEqual(conv.padding, (1, 1, 1, 1))
        self.assertEqual(graph.values[conv.input].layout, FloatLayout.NCHW)
        self.assertEqual(graph.values[conv.weight].layout, FloatLayout.OIHW)
        self.assertEqual(graph.values[conv.weight].kind, FloatValueKind.PARAMETER)
        self.assertEqual(graph.values[conv.weight].source_name, "0.weight")
        linear = graph.ops[4]
        self.assertEqual(graph.values[linear.weight].layout, FloatLayout.OI)
        self.assertEqual(graph.values[graph.inputs[0]].shape, (1, 3, 8, 8))
        self.assertEqual(graph.values[graph.outputs[0]].shape, (1, 5))
        with self.assertRaises(TypeError):
            graph.values["new"] = graph.values[graph.inputs[0]]  # type: ignore[index]
        with self.assertRaises(ValueError):
            graph.constants[conv.weight][0, 0, 0, 0] = 99.0

        second = capture_torch_export(model, torch.randn(1, 3, 8, 8), name="tiny_cnn")
        self.assertEqual(graph.ops, second.ops)
        self.assertEqual(graph.values, second.values)
        for constant_name in graph.constants:
            np.testing.assert_array_equal(graph.constants[constant_name], second.constants[constant_name])

    def test_depthwise_residual_mul_relu6_avgpool_and_parameter_buffer_identification(self) -> None:
        class DepthwiseResidual(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.depthwise = nn.Conv2d(4, 4, 3, padding=1, groups=4)
                self.register_buffer("factor", torch.ones(1, 4, 8, 8))
                self.pool = nn.AvgPool2d(2, count_include_pad=False)

            def forward(self, x):  # type: ignore[no-untyped-def]
                y = functional.relu6(self.depthwise(x))
                return self.pool((y + x) * self.factor)

        graph = capture_torch_export(DepthwiseResidual().eval(), torch.randn(1, 4, 8, 8))
        self.assertEqual(
            tuple(type(op) for op in graph.ops),
            (
                FloatDepthwiseConv2DOp,
                FloatReLU6Op,
                FloatAddOp,
                FloatMulOp,
                FloatAveragePool2DOp,
            ),
        )
        depthwise = graph.ops[0]
        self.assertEqual(depthwise.input_channels, 4)
        self.assertEqual(depthwise.depth_multiplier, 1)
        factor_names = [
            name for name, value in graph.values.items() if value.source_name == "factor"
        ]
        self.assertEqual(len(factor_names), 1)
        self.assertEqual(graph.values[factor_names[0]].kind, FloatValueKind.BUFFER)

    def test_cat_and_reshape_are_typed_and_static(self) -> None:
        class CatReshape(nn.Module):
            def forward(self, x):  # type: ignore[no-untyped-def]
                return torch.cat((x, x), dim=1).reshape(1, -1)

        graph = capture_torch_export(CatReshape().eval(), torch.randn(1, 2, 3, 4))
        self.assertEqual(tuple(type(op) for op in graph.ops), (FloatConcatOp, FloatReshapeOp))
        concat, reshape = graph.ops
        self.assertEqual(concat.axis, 1)
        self.assertEqual(concat.inputs, (graph.inputs[0], graph.inputs[0]))
        self.assertEqual(reshape.target_shape, (1, 48))

    def test_single_input_cat_is_canonicalized_to_an_identity_edge(self) -> None:
        class SingleCat(nn.Module):
            def forward(self, x):  # type: ignore[no-untyped-def]
                return torch.cat([x], dim=1).relu()

        graph = capture_torch_export(SingleCat().eval(), torch.randn(1, 2, 3, 4))
        self.assertEqual(tuple(type(op) for op in graph.ops), (FloatReLUOp,))
        self.assertEqual(graph.ops[0].input, graph.inputs[0])

    def test_eval_conv_batchnorm_relu_is_folded_into_new_immutable_constants(self) -> None:
        conv = nn.Conv2d(2, 2, 1, bias=True)
        batch_norm = nn.BatchNorm2d(2, eps=0.25)
        model = nn.Sequential(conv, batch_norm, nn.ReLU()).eval()
        with torch.no_grad():
            conv.weight.copy_(torch.tensor([[[[1.0]], [[-2.0]]], [[[0.5]], [[3.0]]]]))
            conv.bias.copy_(torch.tensor([0.75, -1.25]))
            batch_norm.weight.copy_(torch.tensor([1.5, -0.5]))
            batch_norm.bias.copy_(torch.tensor([0.2, 2.0]))
            batch_norm.running_mean.copy_(torch.tensor([0.5, -1.0]))
            batch_norm.running_var.copy_(torch.tensor([3.75, 0.75]))

        example = torch.tensor(
            [[[[1.0, -2.0], [0.5, 3.0]], [[-1.0, 2.0], [4.0, -0.5]]]]
        )
        expected = model(example)
        graph = capture_torch_export(model, example, name="conv_bn_relu")
        self.assertEqual(tuple(type(op) for op in graph.ops), (FloatConv2DOp, FloatReLUOp))
        folded_conv, relu = graph.ops
        self.assertEqual(folded_conv.output, relu.input)
        self.assertIn("batch_norm", folded_conv.output)
        self.assertIn("folded_bn", folded_conv.weight)
        self.assertIn("folded_bn", folded_conv.bias)
        self.assertEqual(graph.values[folded_conv.weight].kind, FloatValueKind.CONSTANT)
        self.assertEqual(graph.values[folded_conv.bias].kind, FloatValueKind.CONSTANT)

        original_weight = conv.weight.detach().cpu().numpy()
        original_bias = conv.bias.detach().cpu().numpy()
        gamma = batch_norm.weight.detach().cpu().numpy()
        beta = batch_norm.bias.detach().cpu().numpy()
        mean = batch_norm.running_mean.detach().cpu().numpy()
        variance = batch_norm.running_var.detach().cpu().numpy()
        scale = gamma / np.sqrt(variance + batch_norm.eps)
        np.testing.assert_allclose(
            graph.constants[folded_conv.weight],
            original_weight * scale.reshape(2, 1, 1, 1),
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            graph.constants[folded_conv.bias],
            (original_bias - mean) * scale + beta,
            rtol=1e-6,
            atol=1e-7,
        )
        with self.assertRaises(ValueError):
            graph.constants[folded_conv.weight][0, 0, 0, 0] = 0.0

        folded_output = functional.conv2d(
            example,
            torch.from_numpy(graph.constants[folded_conv.weight].copy()),
            torch.from_numpy(graph.constants[folded_conv.bias].copy()),
        ).relu()
        torch.testing.assert_close(folded_output, expected, rtol=1e-5, atol=1e-6)

    def test_batchnorm_folding_supports_no_original_bias_and_no_affine(self) -> None:
        conv = nn.Conv2d(2, 3, 1, bias=False)
        batch_norm = nn.BatchNorm2d(3, affine=False, eps=0.125)
        model = nn.Sequential(conv, batch_norm).eval()
        with torch.no_grad():
            conv.weight.copy_(torch.arange(6, dtype=torch.float32).reshape(3, 2, 1, 1) - 2.0)
            batch_norm.running_mean.copy_(torch.tensor([1.0, -2.0, 0.5]))
            batch_norm.running_var.copy_(torch.tensor([0.875, 1.875, 3.875]))
        graph = capture_torch_export(model, torch.randn(1, 2, 2, 2))
        self.assertEqual(len(graph.ops), 1)
        folded = graph.ops[0]
        self.assertIsInstance(folded, FloatConv2DOp)
        self.assertIsNotNone(folded.bias)
        scale = 1.0 / np.sqrt(batch_norm.running_var.detach().numpy() + batch_norm.eps)
        np.testing.assert_allclose(
            graph.constants[folded.weight],
            conv.weight.detach().numpy() * scale.reshape(3, 1, 1, 1),
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            graph.constants[folded.bias],
            -batch_norm.running_mean.detach().numpy() * scale,
            rtol=1e-6,
            atol=1e-7,
        )

        linear = nn.Linear(4, 3, bias=False)
        linear_bn = nn.BatchNorm1d(3, affine=False)
        linear_model = nn.Sequential(linear, linear_bn).eval()
        linear_graph = capture_torch_export(linear_model, torch.randn(1, 4))
        self.assertEqual(len(linear_graph.ops), 1)
        self.assertIsInstance(linear_graph.ops[0], FloatLinearOp)
        self.assertIsNotNone(linear_graph.ops[0].bias)

        depthwise = nn.Conv2d(2, 4, 1, groups=2, bias=False)
        depthwise_bn = nn.BatchNorm2d(4, affine=False)
        depthwise_model = nn.Sequential(depthwise, depthwise_bn).eval()
        depthwise_graph = capture_torch_export(
            depthwise_model, torch.randn(1, 2, 2, 2)
        )
        self.assertEqual(len(depthwise_graph.ops), 1)
        self.assertIsInstance(depthwise_graph.ops[0], FloatDepthwiseConv2DOp)
        self.assertIsNotNone(depthwise_graph.ops[0].bias)

    def test_batchnorm_training_shared_producer_and_used_non_fp32_constant_fail_closed(self) -> None:
        class TrainingBatchNorm(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(2, 2, 1)
                self.register_buffer("mean", torch.zeros(2))
                self.register_buffer("variance", torch.ones(2))

            def forward(self, x):  # type: ignore[no-untyped-def]
                y = self.conv(x)
                return torch.ops.aten.batch_norm.default(
                    y, None, None, self.mean, self.variance, True, 0.1, 1e-5, True
                )

        with self.assertRaisesRegex(CompileError, "training=False"):
            capture_torch_export(TrainingBatchNorm().eval(), torch.randn(1, 2, 2, 2))

        class SharedProducer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(2, 2, 1)
                self.batch_norm = nn.BatchNorm2d(2)

            def forward(self, x):  # type: ignore[no-untyped-def]
                y = self.conv(x)
                return self.batch_norm(y) + y

        with self.assertRaisesRegex(CompileError, "shared/fan-out"):
            capture_torch_export(SharedProducer().eval(), torch.randn(1, 2, 2, 2))

        non_finite_eps = nn.Sequential(nn.Conv2d(2, 2, 1), nn.BatchNorm2d(2)).eval()
        non_finite_eps[1].eps = float("inf")
        with self.assertRaisesRegex(CompileError, "eps must be a finite positive real"):
            capture_torch_export(non_finite_eps, torch.randn(1, 2, 2, 2))

        class UsedIntegerBuffer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("integer", torch.ones(1, 4, dtype=torch.int64))

            def forward(self, x):  # type: ignore[no-untyped-def]
                return x + self.integer

        with self.assertRaisesRegex(CompileError, "constants must be float32"):
            capture_torch_export(UsedIntegerBuffer().eval(), torch.randn(1, 4))

    def test_inplace_relu_and_real_relu6_surfaces_capture_without_input_mutation(self) -> None:
        source = torch.tensor([[-3.0, 0.5, 8.0]], dtype=torch.float32)
        relu_model = nn.ReLU(inplace=True).eval()
        relu_graph = capture_torch_export(relu_model, source.clone())
        self.assertEqual(len(relu_graph.ops), 1)
        self.assertIsInstance(relu_graph.ops[0], FloatReLUOp)
        self.assertEqual(relu_graph.ops[0].input, relu_graph.inputs[0])
        self.assertNotEqual(relu_graph.ops[0].output, relu_graph.inputs[0])
        torch.testing.assert_close(relu_model(source.clone()), torch.relu(source))

        for inplace in (False, True):
            with self.subTest(inplace=inplace):
                model = nn.ReLU6(inplace=inplace).eval()
                graph = capture_torch_export(model, source.clone())
                self.assertEqual(len(graph.ops), 1)
                self.assertIsInstance(graph.ops[0], FloatReLU6Op)
                torch.testing.assert_close(model(source.clone()), torch.clamp(source, 0.0, 6.0))

    def test_general_hardtanh_and_unsafe_inplace_fanout_are_rejected(self) -> None:
        with self.assertRaisesRegex(CompileError, "Hardtanh is supported only"):
            capture_torch_export(nn.Hardtanh(-1.0, 1.0).eval(), torch.randn(1, 4))

        class SharedInplaceInput(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("constant", torch.ones(1, 4))

            def forward(self, x):  # type: ignore[no-untyped-def]
                value = x + self.constant
                before_mutation = value + self.constant
                mutated = torch.ops.aten.relu_.default(value)
                return before_mutation + mutated

        with self.assertRaisesRegex(CompileError, "shared/fan-out; mutation semantics"):
            capture_torch_export(SharedInplaceInput().eval(), torch.randn(1, 4))

    def test_adaptive_global_average_pool_is_canonicalized_with_parity(self) -> None:
        source = torch.arange(30, dtype=torch.float32).reshape(1, 2, 3, 5) - 7.0
        model = nn.AdaptiveAvgPool2d((1, 1)).eval()
        expected = model(source)
        graph = capture_torch_export(model, source)
        self.assertEqual(len(graph.ops), 1)
        pool = graph.ops[0]
        self.assertIsInstance(pool, FloatAveragePool2DOp)
        self.assertEqual(pool.kernel, (3, 5))
        self.assertEqual(pool.stride, (3, 5))
        self.assertEqual(pool.padding, (0, 0, 0, 0))
        torch.testing.assert_close(
            functional.avg_pool2d(source, kernel_size=(3, 5), stride=(3, 5)), expected
        )

        with self.assertRaisesRegex(CompileError, r"global output_size=\(1, 1\) only"):
            capture_torch_export(nn.AdaptiveAvgPool2d((2, 1)).eval(), source)

    def test_eval_dropout_and_identity_are_removed_and_training_dropout_is_rejected(self) -> None:
        model = nn.Sequential(nn.Linear(4, 3), nn.Dropout(0.75), nn.ReLU()).eval()
        source = torch.randn(1, 4)
        expected = model(source)
        graph = capture_torch_export(model, source)
        self.assertEqual(tuple(type(op) for op in graph.ops), (FloatLinearOp, FloatReLUOp))
        self.assertEqual(graph.ops[1].input, graph.ops[0].output)
        self.assertFalse(any("dropout" in name for name in graph.values))
        folded = functional.linear(
            source,
            torch.from_numpy(graph.constants[graph.ops[0].weight].copy()),
            torch.from_numpy(graph.constants[graph.ops[0].bias].copy()),
        ).relu()
        torch.testing.assert_close(folded, expected)

        dropout_only = capture_torch_export(nn.Dropout(0.5).eval(), source)
        self.assertEqual(dropout_only.ops, ())
        self.assertEqual(dropout_only.outputs, dropout_only.inputs)
        identity = capture_torch_export(nn.Identity().eval(), source)
        self.assertEqual(identity.ops, ())
        self.assertEqual(identity.outputs, identity.inputs)

        class TrainingDropout(nn.Module):
            def forward(self, x):  # type: ignore[no-untyped-def]
                return torch.ops.aten.dropout.default(x, 0.5, True)

        with self.assertRaisesRegex(CompileError, "Dropout requires training=False"):
            capture_torch_export(TrainingDropout().eval(), source)

    def test_safe_inplace_add_is_normalized_to_immutable_float_add(self) -> None:
        class SafeInplaceAdd(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("constant", torch.ones(1, 4))

            def forward(self, x):  # type: ignore[no-untyped-def]
                value = x + self.constant
                return torch.ops.aten.add_.Tensor(value, self.constant)

        source = torch.tensor([[-2.0, 0.0, 3.0, 7.0]])
        model = SafeInplaceAdd().eval()
        expected = model(source.clone())
        graph = capture_torch_export(model, source.clone())
        self.assertEqual(tuple(type(op) for op in graph.ops), (FloatAddOp, FloatAddOp))
        normalized = graph.ops[-1]
        self.assertEqual(normalized.input_a, graph.ops[0].output)
        self.assertNotEqual(normalized.output, normalized.input_a)
        constant = torch.from_numpy(graph.constants[normalized.input_b].copy())
        captured = (source + constant) + constant
        torch.testing.assert_close(captured, expected)

    def test_inplace_add_shared_target_and_exported_caller_mutation_fail_closed(self) -> None:
        class SharedTarget(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("constant", torch.ones(1, 4))

            def forward(self, x):  # type: ignore[no-untyped-def]
                value = x + self.constant
                old_value = value * self.constant
                mutated = torch.ops.aten.add_.Tensor(value, self.constant)
                return old_value + mutated

        with self.assertRaisesRegex(
            CompileError,
            "aten.add_\\.Tensor target input is shared/fan-out; mutation semantics are unsupported",
        ):
            capture_torch_export(SharedTarget().eval(), torch.randn(1, 4))

        class CallerMutation(nn.Module):
            def forward(self, x):  # type: ignore[no-untyped-def]
                return torch.ops.aten.add_.Tensor(x, x)

        with self.assertRaisesRegex(CompileError, "may not mutate caller or captured constant storage"):
            capture_torch_export(CallerMutation().eval(), torch.randn(1, 4))

    def test_torchvision_resnet18_eval_captures_inplace_residual_adds(self) -> None:
        try:
            import torchvision.models as models
        except (ImportError, RuntimeError) as error:
            self.skipTest(f"torchvision is unavailable: {error}")
        model = models.resnet18(weights=None).eval()
        graph = capture_torch_export(model, torch.randn(1, 3, 224, 224), name="resnet18")
        residual_adds = [op for op in graph.ops if isinstance(op, FloatAddOp)]
        self.assertEqual(len(residual_adds), 8)
        self.assertIsInstance(graph.ops[-1], FloatLinearOp)

    def test_training_dynamic_batch_dtype_and_unsupported_aten_fail_closed(self) -> None:
        model = nn.Sequential(nn.Linear(4, 3))
        with self.assertRaisesRegex(CompileError, "requires eval mode"):
            capture_torch_export(model, torch.randn(1, 4))
        model.eval()
        with self.assertRaisesRegex(CompileError, "dynamic shapes are unsupported"):
            capture_torch_export(model, torch.randn(1, 4), dynamic_shapes={"input": object()})
        with self.assertRaisesRegex(CompileError, "static batch-one"):
            capture_torch_export(model, torch.randn(2, 4))
        with self.assertRaisesRegex(CompileError, "must be float32"):
            capture_torch_export(model.double(), torch.randn(1, 4, dtype=torch.float64))

        class Unsupported(nn.Module):
            def forward(self, x):  # type: ignore[no-untyped-def]
                return torch.sin(x)

        with self.assertRaisesRegex(CompileError, "unsupported torch.export operator aten.sin.default"):
            capture_torch_export(Unsupported().eval(), torch.randn(1, 4))

    def test_static_channel_broadcast_scalar_mul_rejection_and_grouped_conv(self) -> None:
        class BroadcastAdd(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("bias", torch.zeros(1, 4, 1, 1))

            def forward(self, x):  # type: ignore[no-untyped-def]
                return x + self.bias

        broadcast = capture_torch_export(BroadcastAdd().eval(), torch.randn(1, 4, 8, 8))
        self.assertEqual(len(broadcast.ops), 1)
        self.assertIsInstance(broadcast.ops[0], FloatAddOp)

        class ScalarMul(nn.Module):
            def forward(self, x):  # type: ignore[no-untyped-def]
                return x * 2.0

        with self.assertRaisesRegex(CompileError, "scalar/broadcast operands are unsupported"):
            capture_torch_export(ScalarMul().eval(), torch.randn(1, 4))

        grouped = nn.Conv2d(4, 4, 3, padding=1, groups=2).eval()
        captured = capture_torch_export(grouped, torch.randn(1, 4, 8, 8))
        self.assertEqual(len(captured.ops), 1)
        self.assertIsInstance(captured.ops[0], FloatConv2DOp)
        self.assertEqual(captured.ops[0].groups, 2)

    def test_grouped_conv_transpose_and_negative_static_crop_are_normalized(self) -> None:
        class GroupedCrop(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.up = nn.ConvTranspose2d(
                    4, 6, 3, stride=2, padding=1, output_padding=1, groups=2
                )

            def forward(self, value):  # type: ignore[no-untyped-def]
                value = self.up(value)
                return value[:, :, 1:-1, 2:8:2]

        graph = capture_torch_export(GroupedCrop().eval(), torch.randn(1, 4, 4, 5))
        self.assertEqual(
            tuple(type(op) for op in graph.ops),
            (FloatConvTranspose2DOp, FloatSliceOp, FloatSliceOp),
        )
        transpose, crop_height, crop_width = graph.ops
        self.assertEqual(transpose.groups, 2)
        self.assertEqual((crop_height.axis, crop_height.start, crop_height.stop, crop_height.step), (2, 1, 7, 1))
        self.assertEqual((crop_width.axis, crop_width.start, crop_width.stop, crop_width.step), (3, 2, 8, 2))

    def test_wrong_flatten_softmax_and_pooling_semantics_are_rejected(self) -> None:
        class WrongFlatten(nn.Module):
            def forward(self, x):  # type: ignore[no-untyped-def]
                return torch.flatten(x, 0)

        with self.assertRaisesRegex(CompileError, "static batch-one rank-two, rank-three, or rank-four"):
            capture_torch_export(WrongFlatten().eval(), torch.randn(1, 2, 3, 4))

        class WrongSoftmax(nn.Module):
            def forward(self, x):  # type: ignore[no-untyped-def]
                return torch.softmax(x, dim=0)

        with self.assertRaisesRegex(CompileError, "Softmax requires rank-two NC input and final axis"):
            capture_torch_export(WrongSoftmax().eval(), torch.randn(1, 4))

        pool = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=True).eval()
        with self.assertRaisesRegex(CompileError, "excludes padded coordinates"):
            capture_torch_export(pool, torch.randn(1, 2, 4, 4))

    def test_public_target_allowlist_is_exact_and_package_import_is_torch_lazy(self) -> None:
        self.assertEqual(
            set(ALLOWED_ATEN_TARGETS),
            {
                "aten.add_.Tensor",
                "aten.add.Tensor",
                "aten.adaptive_avg_pool2d.default",
                "aten.avg_pool1d.default",
                "aten.avg_pool2d.default",
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
            },
        )
        project_root = Path(__file__).resolve().parents[2]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(project_root / "src")
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import bakenn.frontends.torch_export; "
                    "raise SystemExit(1 if 'torch' in sys.modules else 0)"
                ),
            ],
            env=environment,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())


if __name__ == "__main__":
    unittest.main()
