"""Regressions for model-import boundary and source-semantics audit findings."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import bakenn
from bakenn.errors import CompileError
from bakenn.frontends.tflite import import_tflite
from bakenn.frontends.torch_export import capture_torch_export
from bakenn.ir import (
    DType, Layout, LinearOp, PerAxisQParams, PerTensorQParams,
    QuantizedGraph, ReshapeOp, TensorType,
)


def _identity_mlp() -> QuantizedGraph:
    from benchmarks.tflm_compare.model_fixtures import cmsis_mlp_graph

    graph = cmsis_mlp_graph((4, 4, 4))
    activation = PerTensorQParams(0.25, 0)
    weight = PerAxisQParams((0.25,) * 4, (0,) * 4, 0)
    bias = PerAxisQParams((0.0625,) * 4, (0,) * 4, 0)
    return replace(
        graph,
        values={
            name: replace(value, qparams=(
                bias if value.dtype is DType.INT32
                else weight if value.layout is Layout.OI else activation
            ))
            for name, value in graph.values.items()
        },
        constants={
            "weight_0": np.eye(4, dtype=np.int8) * 4,
            "weight_1": np.eye(4, dtype=np.int8) * 8,
            "bias_0": np.zeros(4, dtype=np.int32),
            "bias_1": np.zeros(4, dtype=np.int32),
        },
        ops=tuple(replace(op, activation_min=-128) for op in graph.ops),
    )


def _assert_litert_and_c(data: bytes, sample: np.ndarray, tmp_path: Path) -> QuantizedGraph:
    litert = pytest.importorskip("ai_edge_litert.interpreter")
    from tests.test_tflite_litert_differential import _c_outputs

    imported = import_tflite(data)
    compiled = bakenn.compile(imported, tmp_path / "generated")
    interpreter = litert.Interpreter(
        model_content=data, num_threads=1,
        experimental_op_resolver_type=litert.OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], sample)
    interpreter.invoke()
    expected = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    np.testing.assert_array_equal(bakenn.run_reference(compiled.plan, sample), expected)
    np.testing.assert_array_equal(_c_outputs(compiled, sample, tmp_path), expected)
    return imported


@pytest.mark.parametrize("synthetic_bias", (False, True))
def test_tflite_tensor_and_synthesized_names_never_alias(
    synthetic_bias: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    pytest.importorskip("tflite")
    import benchmarks.tflm_compare.quantized_graph_to_tflite as adapter

    original_tensor, original_operator, original_opcode = (
        adapter._tensor, adapter._operator, adapter._opcode
    )
    names = (
        {"weight_1": "tflite_0_fully_connected.zero_bias"}
        if synthetic_bias else {"input": "w", "weight_0": "w__4", "weight_1": "w"}
    )

    def renamed_tensor(builder, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["name"] = names.get(kwargs["name"], kwargs["name"])
        return original_tensor(builder, **kwargs)

    def omit_bias(builder, opcode_index, inputs, output, options_type, options):  # type: ignore[no-untyped-def]
        return original_operator(builder, opcode_index, inputs[:2], output, options_type, options)

    def version_six(builder, builtin, version):  # type: ignore[no-untyped-def]
        if builtin == adapter.tflite.BuiltinOperator.FULLY_CONNECTED:
            version = 6
        return original_opcode(builder, builtin, version)

    monkeypatch.setattr(adapter, "_tensor", renamed_tensor)
    if synthetic_bias:
        monkeypatch.setattr(adapter, "_operator", omit_bias)
        monkeypatch.setattr(adapter, "_opcode", version_six)
    data = adapter.export_quantized_graph(_identity_mlp()).data
    imported = _assert_litert_and_c(data, np.array([[4, 8, 12, 16]], dtype=np.int8), tmp_path)
    first, second = imported.ops
    assert first.weight != second.weight
    assert first.bias != second.weight
    np.testing.assert_array_equal(imported.constants[first.weight], np.eye(4, dtype=np.int8) * 4)
    np.testing.assert_array_equal(imported.constants[second.weight], np.eye(4, dtype=np.int8) * 8)


def test_tflite_relu6_uses_float32_rounding_like_litert(tmp_path: Path) -> None:
    pytest.importorskip("tflite")
    from benchmarks.tflm_compare.quantized_graph_to_tflite import export_quantized_graph

    activation = PerTensorQParams(1.0, 0)
    weight = PerAxisQParams((1.0,), (0,), 0)
    output = PerTensorQParams(float(np.float32(2.4)), -128)
    graph = QuantizedGraph(
        "relu6_float32_tie",
        {
            "x": TensorType((1, 1), DType.INT8, Layout.NC, activation),
            "w": TensorType((1, 1), DType.INT8, Layout.OI, weight),
            "b": TensorType((1,), DType.INT32, Layout.C, weight),
            "y": TensorType((1, 1), DType.INT8, Layout.NC, output),
        },
        {"w": np.ones((1, 1), dtype=np.int8), "b": np.zeros(1, dtype=np.int32)},
        (LinearOp("fc", "x", "w", "b", "y", activation_min=-128, activation_max=-126),),
        ("x",), ("y",),
    )
    imported = _assert_litert_and_c(
        export_quantized_graph(graph).data, np.array([[100]], dtype=np.int8), tmp_path,
    )
    assert imported.ops[0].activation_max == -125


def test_tflite_constant_input_reshape_without_options_is_supported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    tflite = pytest.importorskip("tflite")
    import benchmarks.tflm_compare.quantized_graph_to_tflite as adapter

    qparams = PerTensorQParams(0.25, 0)
    graph = QuantizedGraph(
        "reshape_without_options",
        {"x": TensorType((1, 4), DType.INT8, Layout.NC, qparams),
         "y": TensorType((1, 2, 2, 1), DType.INT8, Layout.NHWC, qparams)},
        {}, (ReshapeOp("reshape", "x", "y"),), ("x",), ("y",),
    )

    def omit_options(builder, opcode_index, inputs, output, options_type, options):  # type: ignore[no-untyped-def]
        inputs_vector = adapter._int32_vector(builder, inputs)
        outputs_vector = adapter._int32_vector(builder, (output,))
        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, opcode_index)
        tflite.OperatorAddInputs(builder, inputs_vector)
        tflite.OperatorAddOutputs(builder, outputs_vector)
        return tflite.OperatorEnd(builder)

    monkeypatch.setattr(adapter, "_operator", omit_options)
    _assert_litert_and_c(
        adapter.export_quantized_graph(graph).data,
        np.array([[1, 2, 3, 4]], dtype=np.int8), tmp_path,
    )


def test_tflite_version_six_supports_mixed_present_and_absent_bias(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    tflite = pytest.importorskip("tflite")
    import benchmarks.tflm_compare.quantized_graph_to_tflite as adapter

    original_opcode, original_operator = adapter._opcode, adapter._operator
    calls = 0

    def version_six(builder, builtin, version):  # type: ignore[no-untyped-def]
        return original_opcode(builder, builtin, 6 if builtin == tflite.BuiltinOperator.FULLY_CONNECTED else version)

    def omit_second_bias(builder, opcode_index, inputs, output, options_type, options):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 2:
            inputs = (inputs[0], inputs[1], -1)
        return original_operator(builder, opcode_index, inputs, output, options_type, options)

    monkeypatch.setattr(adapter, "_opcode", version_six)
    monkeypatch.setattr(adapter, "_operator", omit_second_bias)
    _assert_litert_and_c(
        adapter.export_quantized_graph(_identity_mlp()).data,
        np.array([[4, 8, 12, 16]], dtype=np.int8), tmp_path,
    )


@pytest.mark.parametrize("option", ("dilation", "ceil_mode"))
def test_torch_maxpool1d_unsupported_semantics_fail_closed(option: str) -> None:
    torch = pytest.importorskip("torch")

    class Pool(torch.nn.Module):
        def forward(self, value):  # type: ignore[no-untyped-def]
            if option == "dilation":
                return torch.nn.functional.max_pool1d(value, 2, 4, dilation=2)
            return torch.nn.functional.max_pool1d(value, 2, 2, ceil_mode=True)

    with pytest.raises(CompileError, match=option):
        capture_torch_export(Pool().eval(), torch.tensor([[[1.0, 100.0, 3.0, 4.0]]]))


@pytest.mark.parametrize("mutation", ("relu", "add", "silu"))
@pytest.mark.parametrize("alias", ("view", "slice", "dropout"))
def test_torch_mutation_of_shared_alias_fails_closed(mutation: str, alias: str) -> None:
    torch = pytest.importorskip("torch")

    class SharedAlias(torch.nn.Module):
        def forward(self, value):  # type: ignore[no-untyped-def]
            source = value + value
            if alias == "view":
                shared = source.view(source.shape)
            elif alias == "slice":
                shared = torch.ops.aten.slice.Tensor(source, 1, 0, 1)
            else:
                shared = torch.nn.functional.dropout(source, training=False)
            if mutation == "relu":
                changed = shared.relu_()
            elif mutation == "add":
                changed = shared.add_(shared)
            else:
                changed = torch.nn.functional.silu(shared, inplace=True)
            return source + changed

    with pytest.raises(CompileError, match="shared/fan-out"):
        capture_torch_export(SharedAlias().eval(), torch.tensor([[-2.0, 3.0]]))


@pytest.mark.parametrize("alias", ("view", "slice", "dropout"))
def test_torch_mutation_of_unshared_alias_preserves_values(alias: str) -> None:
    torch = pytest.importorskip("torch")
    from bakenn.quantization.ptq_graph import _evaluate

    class UnsharedAlias(torch.nn.Module):
        def forward(self, value):  # type: ignore[no-untyped-def]
            source = value + value
            if alias == "view":
                shared = source.view(source.shape)
            elif alias == "slice":
                shared = torch.ops.aten.slice.Tensor(source, 1, 0, 1)
            else:
                shared = torch.nn.functional.dropout(source, training=False)
            return shared.relu_()

    model = UnsharedAlias().eval()
    sample = torch.tensor([[-2.0, 3.0]])
    graph = capture_torch_export(model, sample)
    np.testing.assert_array_equal(_evaluate(graph, sample.numpy())[graph.outputs[0]], model(sample).numpy())
