from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

import bakenn
from bakenn.errors import CompileError
from bakenn.frontends.tflite import import_tflite
from bakenn.ir import (
    AveragePool2DOp,
    Conv2DOp,
    DType,
    Layout,
    PerAxisQParams,
    PerTensorQParams,
    QuantizedGraph,
    TensorType,
)
from bakenn.ir.ops.pool import AVERAGE_POOL_PROFILE_TFLITE_RAW_V1


_HAS_TFLITE = (
    importlib.util.find_spec("flatbuffers") is not None
    and importlib.util.find_spec("tflite") is not None
)
requires_tflite = pytest.mark.skipif(
    not _HAS_TFLITE, reason="optional flatbuffers/tflite host packages are not installed"
)


def _prefix(graph: QuantizedGraph, count: int) -> QuantizedGraph:
    ops = graph.ops[:count]
    required = set(graph.inputs)
    for op in ops:
        required.update(op.inputs)
        required.update(op.outputs)
    return QuantizedGraph(
        name=f"{graph.name}_prefix_{count}",
        values={name: value for name, value in graph.values.items() if name in required},
        constants={name: value for name, value in graph.constants.items() if name in required},
        ops=ops,
        inputs=graph.inputs,
        outputs=ops[-1].outputs,
    )


def _export(graph: QuantizedGraph) -> bytes:
    from benchmarks.tflm_compare.quantized_graph_to_tflite import export_quantized_graph

    return export_quantized_graph(graph).data


def _with_tflite_average_pool_semantics(graph: QuantizedGraph) -> QuantizedGraph:
    """Make comparison-only serialization explicit about TFLite tie rounding."""

    return replace(
        graph,
        ops=tuple(
            replace(op, arithmetic_profile=AVERAGE_POOL_PROFILE_TFLITE_RAW_V1)
            if isinstance(op, AveragePool2DOp)
            else op
            for op in graph.ops
        ),
    )


def _roundtrip_fixtures() -> tuple[QuantizedGraph, ...]:
    from benchmarks.tflm_compare.model_fixtures import cmsis_mlp_graph
    from tests.p0.model_fixtures import (
        mobilenet_v1_graph,
        residual_ds_cnn_graph,
        tiny_cnn_graph,
    )

    return (
        _prefix(tiny_cnn_graph(), 3),
        residual_ds_cnn_graph(),
        _prefix(mobilenet_v1_graph(), 4),
        cmsis_mlp_graph((8, 5, 3)),
    )


def _explicit_padding_graph() -> QuantizedGraph:
    input_qparams = PerTensorQParams(0.25, -5)
    weight_qparams = PerAxisQParams((0.5,), (0,), 0)
    bias_qparams = PerAxisQParams((0.125,), (0,), 0)
    return QuantizedGraph(
        name="tflite_explicit_padding",
        values={
            "input": TensorType((1, 3, 3, 1), DType.INT8, Layout.NHWC, input_qparams),
            "weight": TensorType((1, 2, 2, 1), DType.INT8, Layout.OHWI, weight_qparams),
            "bias": TensorType((1,), DType.INT32, Layout.C, bias_qparams),
            "output": TensorType((1, 3, 3, 1), DType.INT8, Layout.NHWC, input_qparams),
        },
        constants={
            "weight": np.asarray([[[[2], [-3]], [[4], [1]]]], dtype=np.int8),
            "bias": np.asarray([0], dtype=np.int32),
        },
        ops=(
            Conv2DOp(
                "conv",
                "input",
                "weight",
                "bias",
                "output",
                padding=(1, 0, 0, 1),
            ),
        ),
        inputs=("input",),
        outputs=("output",),
    )


def _same_asymmetric_graph() -> QuantizedGraph:
    input_qparams = PerTensorQParams(0.25, -3)
    output_qparams = PerTensorQParams(0.5, 2)
    weight_qparams = PerAxisQParams((0.125,), (0,), 0)
    bias_qparams = PerAxisQParams((0.03125,), (0,), 0)
    return QuantizedGraph(
        name="tflite_same_asymmetric",
        values={
            "input": TensorType((1, 4, 4, 1), DType.INT8, Layout.NHWC, input_qparams),
            "weight": TensorType((1, 3, 3, 1), DType.INT8, Layout.OHWI, weight_qparams),
            "bias": TensorType((1,), DType.INT32, Layout.C, bias_qparams),
            "output": TensorType((1, 2, 2, 1), DType.INT8, Layout.NHWC, output_qparams),
        },
        constants={
            "weight": np.arange(1, 10, dtype=np.int8).reshape((1, 3, 3, 1)),
            "bias": np.asarray([7], dtype=np.int32),
        },
        ops=(
            Conv2DOp(
                "conv",
                "input",
                "weight",
                "bias",
                "output",
                stride=(2, 2),
                padding=(0, 1, 0, 1),
            ),
        ),
        inputs=("input",),
        outputs=("output",),
    )


def _zero_bias_fc_graph() -> QuantizedGraph:
    from benchmarks.tflm_compare.model_fixtures import cmsis_mlp_graph

    graph = cmsis_mlp_graph((4, 3))
    bias_name = graph.ops[0].bias  # type: ignore[attr-defined]
    constants = dict(graph.constants)
    constants[bias_name] = np.zeros((3,), dtype=np.int32)
    return QuantizedGraph(
        graph.name,
        graph.values,
        constants,
        graph.ops,
        graph.inputs,
        graph.outputs,
    )


def test_missing_optional_dependency_is_a_clear_compile_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import bakenn.frontends.tflite.importer as importer

    real_import = importer.importlib.import_module

    def missing(name: str):  # type: ignore[no-untyped-def]
        if name == "flatbuffers":
            return SimpleNamespace()
        if name == "tflite":
            raise ModuleNotFoundError("No module named 'tflite'", name="tflite")
        return real_import(name)

    monkeypatch.setattr(importer.importlib, "import_module", missing)
    with pytest.raises(CompileError, match="optional host packages tflite"):
        import_tflite(b"\0\0\0\0TFL3")


def test_old_tflite_schema_package_is_rejected_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bakenn.frontends.tflite.importer as importer

    real_import = importer.importlib.import_module
    old_schema = SimpleNamespace(Tensor=object)

    def old_version(name: str):  # type: ignore[no-untyped-def]
        if name == "flatbuffers":
            return SimpleNamespace()
        if name == "tflite":
            return old_schema
        return real_import(name)

    monkeypatch.setattr(importer.importlib, "import_module", old_version)
    with pytest.raises(CompileError, match=r"too old.*tflite>=2\.18"):
        import_tflite(b"\0\0\0\0TFL3")


def test_invalid_explicit_graph_name_is_rejected_before_parsing() -> None:
    with pytest.raises(CompileError, match="name must be a non-empty string"):
        import_tflite(b"\0\0\0\0TFL3", name="")


@requires_tflite
def test_rejects_non_tflite_and_truncated_flatbuffers() -> None:
    with pytest.raises(CompileError, match="TFL3 identifier"):
        import_tflite(b"not-a-tflite-model")
    with pytest.raises(CompileError, match="malformed or unsupported"):
        import_tflite(b"\x08\0\0\0TFL3")


@requires_tflite
@pytest.mark.parametrize("fixture_index", range(4))
def test_benchmark_exporter_roundtrips_supported_graphs_byte_exact(
    fixture_index: int, tmp_path: Path
) -> None:
    source = _with_tflite_average_pool_semantics(
        _roundtrip_fixtures()[fixture_index]
    )
    imported = import_tflite(_export(source), name=f"imported_{fixture_index}")
    for op in imported.ops:
        if isinstance(op, AveragePool2DOp):
            assert op.arithmetic_profile == AVERAGE_POOL_PROFILE_TFLITE_RAW_V1
    source_compiled = bakenn.compile(source, tmp_path / "source")
    imported_compiled = bakenn.compile(imported, tmp_path / "imported")
    rng = np.random.default_rng(20260824 + fixture_index)
    input_shape = source.values[source.inputs[0]].shape
    for _ in range(32):
        input_codes = rng.integers(-128, 128, size=input_shape, dtype=np.int16).astype(np.int8)
        expected = bakenn.run_reference(source_compiled.plan, input_codes)
        actual = bakenn.run_reference(imported_compiled.plan, input_codes)
        np.testing.assert_array_equal(actual, expected)


@requires_tflite
def test_frozen_mnist_tflite_import_matches_all_committed_expected_bytes(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    graph = import_tflite(
        root / "benchmarks/microtvm_compare/results/mnist_common_int8.tflite"
    )
    compiled = bakenn.compile(graph, tmp_path / "mnist")
    input_shape = graph.values[graph.inputs[0]].shape
    inputs = np.fromfile(
        root / "examples/mnist/evidence/physical_test_inputs_int8.bin", dtype=np.int8
    ).reshape(100, *input_shape)
    expected = np.fromfile(
        root / "benchmarks/microtvm_compare/results/mnist_common_expected_int8.bin",
        dtype=np.int8,
    ).reshape(100, -1)
    actual = np.concatenate(
        [bakenn.run_reference(compiled.plan, sample).reshape(1, -1) for sample in inputs],
        axis=0,
    )
    np.testing.assert_array_equal(actual, expected)


@requires_tflite
def test_public_one_call_tflite_compilation(tmp_path: Path) -> None:
    source = _export(_roundtrip_fixtures()[3])
    compiled = bakenn.compile_tflite(
        source,
        tmp_path / "one_call",
        name="one_call_tflite",
    )
    assert compiled.plan.name == "one_call_tflite"
    assert compiled.artifacts.header.is_file()
    assert bakenn.load_manifest(compiled.artifacts.manifest)["model"] == (
        "bknn_one_call_tflite"
    )


@requires_tflite
def test_same_padding_recovers_tflite_asymmetric_split() -> None:
    source = _same_asymmetric_graph()
    imported = import_tflite(_export(source))
    op = imported.ops[0]
    assert isinstance(op, Conv2DOp)
    assert op.padding == (0, 1, 0, 1)


@requires_tflite
def test_padv2_real_zero_becomes_static_pad_then_valid_conv(
    tmp_path: Path,
) -> None:
    source = _explicit_padding_graph()
    imported = import_tflite(_export(source))
    assert [type(op).__name__ for op in imported.ops] == ["Pad2DOp", "Conv2DOp"]
    assert imported.ops[0].padding == (1, 0, 0, 1)  # type: ignore[attr-defined]
    assert imported.ops[1].padding == (0, 0, 0, 0)  # type: ignore[attr-defined]
    rng = np.random.default_rng(20260824)
    source_plan = bakenn.compile(source, tmp_path / "source_pad").plan
    imported_plan = bakenn.compile(imported, tmp_path / "imported_pad").plan
    for _ in range(32):
        values = rng.integers(-128, 128, size=(1, 3, 3, 1), dtype=np.int16).astype(np.int8)
        np.testing.assert_array_equal(
            bakenn.run_reference(imported_plan, values),
            bakenn.run_reference(source_plan, values),
        )


@requires_tflite
def test_pad_without_explicit_value_maps_to_affine_real_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import benchmarks.tflm_compare.quantized_graph_to_tflite as adapter

    source = _explicit_padding_graph()
    original_opcode = adapter._opcode
    original_operator = adapter._operator

    def pad_opcode(builder, builtin, version):  # type: ignore[no-untyped-def]
        if builtin == adapter.tflite.BuiltinOperator.PADV2:
            builtin = adapter.tflite.BuiltinOperator.PAD
            version = 2
        return original_opcode(builder, builtin, version)

    def omit_pad_value(builder, opcode_index, inputs, output, options_type, options):  # type: ignore[no-untyped-def]
        if opcode_index == 7:
            inputs = inputs[:2]
            options_type = adapter.tflite.BuiltinOptions.PadOptions
        return original_operator(builder, opcode_index, inputs, output, options_type, options)

    monkeypatch.setattr(adapter.tflite, "PadV2OptionsStart", adapter.tflite.PadOptionsStart)
    monkeypatch.setattr(adapter.tflite, "PadV2OptionsEnd", adapter.tflite.PadOptionsEnd)
    monkeypatch.setattr(adapter, "_opcode", pad_opcode)
    monkeypatch.setattr(adapter, "_operator", omit_pad_value)
    imported = import_tflite(adapter.export_quantized_graph(source).data)
    assert [type(op).__name__ for op in imported.ops] == ["Pad2DOp", "Conv2DOp"]
    assert imported.values[imported.ops[0].input].qparams.zero_point == -5  # type: ignore[attr-defined,union-attr]


@requires_tflite
def test_padv2_rejects_a_value_other_than_affine_real_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import benchmarks.tflm_compare.quantized_graph_to_tflite as adapter

    original = adapter._buffer

    def wrong_pad_value(builder, value):  # type: ignore[no-untyped-def]
        if value is not None and value.dtype == np.int8 and value.shape == (1,):
            value = np.asarray((-4,), dtype=np.int8)
        return original(builder, value)

    monkeypatch.setattr(adapter, "_buffer", wrong_pad_value)
    data = adapter.export_quantized_graph(_explicit_padding_graph()).data
    with pytest.raises(CompileError, match="PADV2 value must equal input zero point -5"):
        import_tflite(data)


@requires_tflite
def test_optional_fully_connected_bias_is_synthesized_as_exact_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import benchmarks.tflm_compare.quantized_graph_to_tflite as adapter

    source = _zero_bias_fc_graph()
    original = adapter._operator
    original_opcode = adapter._opcode

    def omit_bias(builder, opcode_index, inputs, output, options_type, options):  # type: ignore[no-untyped-def]
        return original(builder, opcode_index, inputs[:2], output, options_type, options)

    def no_bias_version(builder, builtin, version):  # type: ignore[no-untyped-def]
        if builtin == adapter.tflite.BuiltinOperator.FULLY_CONNECTED:
            version = 6
        return original_opcode(builder, builtin, version)

    monkeypatch.setattr(adapter, "_operator", omit_bias)
    monkeypatch.setattr(adapter, "_opcode", no_bias_version)
    imported = import_tflite(adapter.export_quantized_graph(source).data)
    bias_name = imported.ops[0].bias  # type: ignore[attr-defined]
    assert bias_name.endswith(".zero_bias")
    np.testing.assert_array_equal(imported.constants[bias_name], np.zeros((3,), dtype=np.int32))


@requires_tflite
def test_relu6_fused_activation_is_imported_as_quantized_clamp() -> None:
    source = _same_asymmetric_graph()
    relu6_op = replace(source.ops[0], activation_min=2, activation_max=14)
    source = replace(source, ops=(relu6_op,))
    imported = import_tflite(_export(source))
    assert imported.ops[0].activation_min == 2  # type: ignore[attr-defined]
    assert imported.ops[0].activation_max == 14  # type: ignore[attr-defined]


@requires_tflite
def test_unsupported_fused_activation_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import benchmarks.tflm_compare.quantized_graph_to_tflite as adapter

    monkeypatch.setattr(
        adapter,
        "_activation",
        lambda op, output_qparams: adapter.tflite.ActivationFunctionType.TANH,
    )
    data = adapter.export_quantized_graph(_prefix(_roundtrip_fixtures()[0], 1)).data
    with pytest.raises(CompileError, match="only NONE, RELU and RELU6"):
        import_tflite(data)


@requires_tflite
def test_softmax_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import benchmarks.tflm_compare.quantized_graph_to_tflite as adapter

    original = adapter._opcode

    def softmax_opcode(builder, builtin, version):  # type: ignore[no-untyped-def]
        del builtin, version
        return original(builder, adapter.tflite.BuiltinOperator.SOFTMAX, 2)

    monkeypatch.setattr(adapter, "_opcode", softmax_opcode)
    data = adapter.export_quantized_graph(_prefix(_roundtrip_fixtures()[0], 1)).data
    with pytest.raises(CompileError, match="unsupported TFLite builtin operator"):
        import_tflite(data)


@requires_tflite
def test_unsupported_operator_version_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import benchmarks.tflm_compare.quantized_graph_to_tflite as adapter

    original = adapter._opcode

    def unsupported_version(builder, builtin, version):  # type: ignore[no-untyped-def]
        del version
        return original(builder, builtin, 99)

    monkeypatch.setattr(adapter, "_opcode", unsupported_version)
    data = adapter.export_quantized_graph(_prefix(_roundtrip_fixtures()[0], 1)).data
    with pytest.raises(CompileError, match="version 99 is unsupported"):
        import_tflite(data)


@requires_tflite
def test_custom_operator_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import benchmarks.tflm_compare.quantized_graph_to_tflite as adapter

    original = adapter._opcode

    def custom_opcode(builder, builtin, version):  # type: ignore[no-untyped-def]
        del builtin
        return original(builder, adapter.tflite.BuiltinOperator.CUSTOM, version)

    monkeypatch.setattr(adapter, "_opcode", custom_opcode)
    data = adapter.export_quantized_graph(_prefix(_roundtrip_fixtures()[0], 1)).data
    with pytest.raises(CompileError, match="unsupported custom op"):
        import_tflite(data)


@requires_tflite
def test_variable_tensor_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import benchmarks.tflm_compare.quantized_graph_to_tflite as adapter

    original = adapter.tflite.TensorEnd

    def variable_tensor_end(builder):  # type: ignore[no-untyped-def]
        adapter.tflite.TensorAddIsVariable(builder, True)
        return original(builder)

    monkeypatch.setattr(adapter.tflite, "TensorEnd", variable_tensor_end)
    data = adapter.export_quantized_graph(_prefix(_roundtrip_fixtures()[0], 1)).data
    with pytest.raises(CompileError, match="is variable"):
        import_tflite(data)


@requires_tflite
def test_dynamic_shape_signature_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import benchmarks.tflm_compare.quantized_graph_to_tflite as adapter

    def dynamic_tensor(
        builder,
        *,
        name,
        shape,
        tensor_type,
        buffer_index,
        qparams,
        axis_override=None,
    ):  # type: ignore[no-untyped-def]
        name_offset = builder.CreateString(name)
        shape_offset = adapter._int32_vector(builder, shape)
        signature_offset = adapter._int32_vector(builder, (-1, *shape[1:]))
        quantization = adapter._quantization(
            builder, qparams, axis_override=axis_override
        )
        adapter.tflite.TensorStart(builder)
        adapter.tflite.TensorAddShape(builder, shape_offset)
        adapter.tflite.TensorAddShapeSignature(builder, signature_offset)
        adapter.tflite.TensorAddType(builder, tensor_type)
        adapter.tflite.TensorAddBuffer(builder, buffer_index)
        adapter.tflite.TensorAddName(builder, name_offset)
        adapter.tflite.TensorAddQuantization(builder, quantization)
        return adapter.tflite.TensorEnd(builder)

    monkeypatch.setattr(adapter, "_tensor", dynamic_tensor)
    data = adapter.export_quantized_graph(_prefix(_roundtrip_fixtures()[0], 1)).data
    with pytest.raises(CompileError, match="dynamic/incompatible shape signature"):
        import_tflite(data)


@requires_tflite
def test_sparse_tensor_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import benchmarks.tflm_compare.quantized_graph_to_tflite as adapter

    def sparse_tensor(
        builder,
        *,
        name,
        shape,
        tensor_type,
        buffer_index,
        qparams,
        axis_override=None,
    ):  # type: ignore[no-untyped-def]
        name_offset = builder.CreateString(name)
        shape_offset = adapter._int32_vector(builder, shape)
        quantization = adapter._quantization(
            builder, qparams, axis_override=axis_override
        )
        adapter.tflite.SparsityParametersStart(builder)
        sparsity = adapter.tflite.SparsityParametersEnd(builder)
        adapter.tflite.TensorStart(builder)
        adapter.tflite.TensorAddShape(builder, shape_offset)
        adapter.tflite.TensorAddType(builder, tensor_type)
        adapter.tflite.TensorAddBuffer(builder, buffer_index)
        adapter.tflite.TensorAddName(builder, name_offset)
        adapter.tflite.TensorAddQuantization(builder, quantization)
        adapter.tflite.TensorAddSparsity(builder, sparsity)
        return adapter.tflite.TensorEnd(builder)

    monkeypatch.setattr(adapter, "_tensor", sparse_tensor)
    data = adapter.export_quantized_graph(_prefix(_roundtrip_fixtures()[0], 1)).data
    with pytest.raises(CompileError, match="unsupported sparse storage"):
        import_tflite(data)


@requires_tflite
def test_wrong_per_axis_quantized_dimension_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import benchmarks.tflm_compare.quantized_graph_to_tflite as adapter

    original = adapter.tflite.QuantizationParametersAddQuantizedDimension

    def wrong_axis(builder, axis):  # type: ignore[no-untyped-def]
        return original(builder, 1 if axis == 0 else axis)

    monkeypatch.setattr(
        adapter.tflite, "QuantizationParametersAddQuantizedDimension", wrong_axis
    )
    data = adapter.export_quantized_graph(_prefix(_roundtrip_fixtures()[0], 1)).data
    with pytest.raises(CompileError, match="quantized_dimension must be 0"):
        import_tflite(data)


@requires_tflite
def test_nonzero_weight_zero_point_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import benchmarks.tflm_compare.quantized_graph_to_tflite as adapter

    original = adapter._tensor

    def asymmetric_weight(
        builder,
        *,
        name,
        shape,
        tensor_type,
        buffer_index,
        qparams,
        axis_override=None,
    ):  # type: ignore[no-untyped-def]
        if name == "conv.weight":
            assert isinstance(qparams, PerAxisQParams)
            qparams = PerAxisQParams(
                qparams.scales, (1,) * len(qparams.zero_points), qparams.axis
            )
        return original(
            builder,
            name=name,
            shape=shape,
            tensor_type=tensor_type,
            buffer_index=buffer_index,
            qparams=qparams,
            axis_override=axis_override,
        )

    monkeypatch.setattr(adapter, "_tensor", asymmetric_weight)
    data = adapter.export_quantized_graph(_prefix(_roundtrip_fixtures()[0], 1)).data
    with pytest.raises(CompileError, match="weight conv.weight must have zero point 0"):
        import_tflite(data)


@requires_tflite
def test_bias_scale_must_equal_input_times_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import benchmarks.tflm_compare.quantized_graph_to_tflite as adapter

    original = adapter._tensor

    def wrong_bias_scale(
        builder,
        *,
        name,
        shape,
        tensor_type,
        buffer_index,
        qparams,
        axis_override=None,
    ):  # type: ignore[no-untyped-def]
        if name == "conv.bias":
            assert isinstance(qparams, PerAxisQParams)
            qparams = PerAxisQParams(
                tuple(scale * 2.0 for scale in qparams.scales),
                qparams.zero_points,
                qparams.axis,
            )
        return original(
            builder,
            name=name,
            shape=shape,
            tensor_type=tensor_type,
            buffer_index=buffer_index,
            qparams=qparams,
            axis_override=axis_override,
        )

    monkeypatch.setattr(adapter, "_tensor", wrong_bias_scale)
    data = adapter.export_quantized_graph(_prefix(_roundtrip_fixtures()[0], 1)).data
    with pytest.raises(CompileError, match=r"scale must equal input_scale \* weight_scale"):
        import_tflite(data)


@requires_tflite
def test_imported_graph_generates_strict_c_matching_python(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("host C compiler is unavailable")
    source = _roundtrip_fixtures()[1]
    imported = import_tflite(_export(source), name="tflite_generated_c")
    compiled = bakenn.compile(imported, tmp_path / "generated")
    artifacts = compiled.artifacts
    symbol = "bknn_tflite_generated_c"
    macro = symbol.upper()
    runner = tmp_path / "generated" / "runner.c"
    runner.write_text(
        f'''#include "{artifacts.header.name}"
#include <stdio.h>

int main(void) {{
    _Alignas({macro}_ARENA_ALIGNMENT) unsigned char arena[{macro}_ARENA_SIZE];
    signed char input[{macro}_INPUT_SIZE];
    signed char output[{macro}_OUTPUT_SIZE];
    while (fread(input, 1u, {macro}_INPUT_SIZE, stdin) == {macro}_INPUT_SIZE) {{
        {symbol}_infer(arena, input, output);
        if (fwrite(output, 1u, {macro}_OUTPUT_SIZE, stdout) != {macro}_OUTPUT_SIZE) {{
            return 2;
        }}
    }}
    return ferror(stdin) ? 3 : 0;
}}
''',
        encoding="utf-8",
    )
    executable = tmp_path / "generated" / "runner"
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-O1",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            str(artifacts.model_source),
            str(artifacts.weights_source),
            str(artifacts.kernels_source),
            str(runner),
            "-I",
            str(artifacts.output_dir),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
    )
    rng = np.random.default_rng(20260824)
    input_shape = imported.values[imported.inputs[0]].shape
    corpus = rng.integers(-128, 128, size=(64, *input_shape[1:]), dtype=np.int16).astype(
        np.int8
    )
    expected = np.concatenate(
        [
            bakenn.run_reference(compiled.plan, sample.reshape(input_shape))
            for sample in corpus
        ],
        axis=0,
    )
    result = subprocess.run(executable, input=corpus.tobytes(), capture_output=True, check=True)
    actual = np.frombuffer(result.stdout, dtype=np.int8).reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected)
