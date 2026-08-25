from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

litert = pytest.importorskip("ai_edge_litert.interpreter")
pytest.importorskip("flatbuffers")
pytest.importorskip("tflite")

import bakenn
from bakenn.frontends.tflite import import_tflite
from bakenn.ir import (
    AveragePool2DOp,
    DType,
    Layout,
    PerTensorQParams,
    QuantizedGraph,
    TensorType,
)
from bakenn.ir.ops.pool import AVERAGE_POOL_PROFILE_TFLITE_RAW_V1
from benchmarks.tflm_compare.model_fixtures import cmsis_mlp_graph
from benchmarks.tflm_compare.quantized_graph_to_tflite import export_quantized_graph
from tests.p0.model_fixtures import (
    mobilenet_v1_graph,
    residual_ds_cnn_graph,
    tiny_cnn_graph,
)


def _prefix(graph: QuantizedGraph, count: int) -> QuantizedGraph:
    ops = graph.ops[:count]
    required = set(graph.inputs)
    for op in ops:
        required.update(op.inputs)
        required.update(op.outputs)
    return QuantizedGraph(
        name=f"{graph.name}_litert_prefix_{count}",
        values={name: value for name, value in graph.values.items() if name in required},
        constants={name: value for name, value in graph.constants.items() if name in required},
        ops=ops,
        inputs=graph.inputs,
        outputs=ops[-1].outputs,
    )


def _average_pool_tie_graph() -> QuantizedGraph:
    qparams = PerTensorQParams(1.0, 1)
    return QuantizedGraph(
        name="litert_average_pool_raw_ties",
        values={
            "input": TensorType((1, 1, 2, 1), DType.INT8, Layout.NHWC, qparams),
            "output": TensorType((1, 1, 1, 1), DType.INT8, Layout.NHWC, qparams),
        },
        constants={},
        ops=(
            AveragePool2DOp(
                "average",
                "input",
                "output",
                kernel=(1, 2),
                stride=(1, 2),
                arithmetic_profile=AVERAGE_POOL_PROFILE_TFLITE_RAW_V1,
            ),
        ),
        inputs=("input",),
        outputs=("output",),
    )


def _c_outputs(
    compiled: bakenn.CompiledModel,
    corpus: np.ndarray,
    tmp_path: Path,
) -> np.ndarray:
    cc = shutil.which("cc")
    if cc is None:
        pytest.fail("a host C compiler is required for the LiteRT differential gate")
    artifacts = compiled.artifacts
    manifest = bakenn.load_manifest(artifacts.manifest)
    symbol = str(manifest["model"])
    macro = symbol.upper()
    runner = tmp_path / "runner.c"
    runner.write_text(
        f'''#include "{artifacts.header.name}"
#include <stdio.h>

int main(void) {{
    _Alignas({macro}_ARENA_ALIGNMENT)
        unsigned char arena[{macro}_ARENA_SIZE > 0u ? {macro}_ARENA_SIZE : 1u];
    signed char input[{macro}_INPUT_SIZE];
    signed char output[{macro}_OUTPUT_SIZE];
    while (fread(input, 1u, {macro}_INPUT_SIZE, stdin) == {macro}_INPUT_SIZE) {{
        {symbol}_infer({macro}_ARENA_SIZE > 0u ? arena : NULL, input, output);
        if (fwrite(output, 1u, {macro}_OUTPUT_SIZE, stdout) != {macro}_OUTPUT_SIZE) {{
            return 2;
        }}
    }}
    return ferror(stdin) ? 3 : 0;
}}
''',
        encoding="utf-8",
    )
    executable = tmp_path / "runner"
    subprocess.run(
        [
            cc,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-I",
            str(artifacts.output_dir),
            str(artifacts.model_source),
            str(artifacts.weights_source),
            str(artifacts.kernels_source),
            str(runner),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
    )
    completed = subprocess.run(
        [str(executable)], input=corpus.tobytes(), check=True, capture_output=True
    )
    output_shape = compiled.plan.tensors[compiled.plan.outputs[0]].tensor_type.shape
    return np.frombuffer(completed.stdout, dtype=np.int8).reshape(
        corpus.shape[0], *output_shape[1:]
    )


@pytest.mark.parametrize("family", ("mlp", "cnn", "residual", "mobilenet"))
def test_litert_imported_reference_and_generated_c_are_byte_exact(
    family: str, tmp_path: Path
) -> None:
    if family == "mlp":
        source = cmsis_mlp_graph((32, 16, 4))
    elif family == "cnn":
        source = _prefix(tiny_cnn_graph(), 3)
    elif family == "residual":
        source = residual_ds_cnn_graph()
    else:
        source = _prefix(mobilenet_v1_graph(), 4)
        source = replace(
            source,
            ops=tuple(
                replace(op, arithmetic_profile=AVERAGE_POOL_PROFILE_TFLITE_RAW_V1)
                if isinstance(op, AveragePool2DOp)
                else op
                for op in source.ops
            ),
        )
    exported = export_quantized_graph(source)
    imported = import_tflite(exported.data, name=f"litert_{family}")
    compiled = bakenn.compile(imported, tmp_path / "generated")
    input_shape = imported.values[imported.inputs[0]].shape
    output_shape = imported.values[imported.outputs[0]].shape
    # Seed 1202 / item 105 is a frozen MobileNet AveragePool tie regression.
    seed = 1202 if family == "mobilenet" else 0xBACE00 + len(source.ops)
    corpus_size = 128 if family == "mobilenet" else 32
    rng = np.random.default_rng(seed)
    corpus = rng.integers(
        -128,
        128,
        size=(corpus_size, *input_shape[1:]),
        dtype=np.int16,
    ).astype(np.int8)

    if family == "mobilenet":
        pre_pool = _prefix(mobilenet_v1_graph(), 2)
        pre_pool_plan = bakenn.compile(
            pre_pool, tmp_path / "mobilenet_pre_pool"
        ).plan
        pre_pool_output = bakenn.run_reference(
            pre_pool_plan, corpus[105].reshape(input_shape)
        )
        raw_sums = pre_pool_output.astype(np.int32).sum(axis=(1, 2)).reshape(-1)
        np.testing.assert_array_equal(raw_sums, np.asarray((874, -8, 1180)))

    # The importer maps each versioned TFLite integer contract to an explicit
    # BakeNN step profile. Host delegates such as XNNPACK are not used as the
    # firmware correctness oracle; BUILTIN_REF is.
    interpreter = litert.Interpreter(
        model_content=exported.data,
        num_threads=1,
        experimental_op_resolver_type=litert.OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    litert_outputs: list[np.ndarray] = []
    reference_outputs: list[np.ndarray] = []
    for sample in corpus:
        model_input = sample.reshape(input_shape)
        interpreter.set_tensor(input_detail["index"], model_input)
        interpreter.invoke()
        litert_outputs.append(
            np.array(interpreter.get_tensor(output_detail["index"]), copy=True)
        )
        reference_outputs.append(
            bakenn.run_reference(compiled.plan, model_input)
        )
    expected = np.concatenate(litert_outputs, axis=0).reshape(
        corpus.shape[0], *output_shape[1:]
    )
    reference = np.concatenate(reference_outputs, axis=0).reshape(expected.shape)
    generated_c = _c_outputs(compiled, corpus, tmp_path)

    if family == "mobilenet":
        # The middle channel is raw sum -8 over 16 positions: TFLite rounds
        # -0.5 to -1. The legacy centered-zp profile would produce code 0.
        np.testing.assert_array_equal(expected[105], np.asarray((55, -1, 74)))
    np.testing.assert_array_equal(reference, expected)
    np.testing.assert_array_equal(generated_c, expected)


def test_litert_average_pool_raw_code_ties_match_python_and_generated_c(
    tmp_path: Path,
) -> None:
    exported = export_quantized_graph(_average_pool_tie_graph())
    imported = import_tflite(exported.data, name="litert_average_pool_ties")
    assert isinstance(imported.ops[0], AveragePool2DOp)
    assert imported.ops[0].arithmetic_profile == AVERAGE_POOL_PROFILE_TFLITE_RAW_V1
    compiled = bakenn.compile(imported, tmp_path / "generated")
    assert compiled.plan.steps[0].arithmetic_profile == AVERAGE_POOL_PROFILE_TFLITE_RAW_V1

    input_shape = imported.values[imported.inputs[0]].shape
    output_shape = imported.values[imported.outputs[0]].shape
    rng = np.random.default_rng(20260824)
    corpus = rng.integers(
        -128,
        128,
        size=(128, *input_shape[1:]),
        dtype=np.int16,
    ).astype(np.int8)
    # Official TFLite raw-code rounding gives 1 here. Centering around zp=1
    # before the half-away division would instead give 0.
    corpus[0] = np.asarray([[[0], [1]]], dtype=np.int8)

    interpreter = litert.Interpreter(
        model_content=exported.data,
        num_threads=1,
        experimental_op_resolver_type=litert.OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    litert_outputs: list[np.ndarray] = []
    reference_outputs: list[np.ndarray] = []
    for sample in corpus:
        model_input = sample.reshape(input_shape)
        interpreter.set_tensor(input_detail["index"], model_input)
        interpreter.invoke()
        litert_outputs.append(
            np.array(interpreter.get_tensor(output_detail["index"]), copy=True)
        )
        reference_outputs.append(bakenn.run_reference(compiled.plan, model_input))

    expected = np.concatenate(litert_outputs, axis=0).reshape(
        corpus.shape[0], *output_shape[1:]
    )
    reference = np.concatenate(reference_outputs, axis=0).reshape(expected.shape)
    generated_c = _c_outputs(compiled, corpus, tmp_path)

    assert int(expected.reshape(corpus.shape[0], -1)[0, 0]) == 1
    np.testing.assert_array_equal(reference, expected)
    np.testing.assert_array_equal(generated_c, expected)
