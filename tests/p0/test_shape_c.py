from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from bakenn.backend.portable_c import generate_portable_c
import bakenn.backend.portable_c.families.shape  # noqa: F401 - installs emitter registrations
from bakenn.errors import CompileError, GraphValidationError
from bakenn.ir import DType, Layout, PerTensorQParams, QuantizedGraph, TensorType, verify_graph
from bakenn.ir.ops.shape import ConcatenateOp, FlattenOp, ReshapeOp, SliceOp
import bakenn.ir.verifiers.shape  # noqa: F401 - installs verifier registrations
from bakenn.plan import Storage, lower_to_plan
import bakenn.plan.lowering.shape  # noqa: F401 - installs lowering registrations
from bakenn.reference import run_reference
import bakenn.reference.kernels.shape  # noqa: F401 - installs reference registrations


QPARAMS = PerTensorQParams(0.25, -7)


def _flatten_concat_graph() -> QuantizedGraph:
    return QuantizedGraph(
        name="flatten_concat",
        values={
            "input": TensorType((1, 1, 2, 2), DType.INT8, Layout.NHWC, QPARAMS),
            "flat": TensorType((1, 4), DType.INT8, Layout.NC, QPARAMS),
            "suffix": TensorType((1, 2), DType.INT8, Layout.NC, QPARAMS),
            "output": TensorType((1, 6), DType.INT8, Layout.NC, QPARAMS),
        },
        constants={"suffix": np.asarray([[91, -92]], dtype=np.int8)},
        ops=(
            FlattenOp("flatten", "input", "flat"),
            ConcatenateOp("concat", ("flat", "suffix"), "output", -1),
        ),
        inputs=("input",),
        outputs=("output",),
    )


def _reshape_concat_graph() -> QuantizedGraph:
    return QuantizedGraph(
        name="reshape_concat",
        values={
            "input": TensorType((1, 4), DType.INT8, Layout.NC, QPARAMS),
            "image": TensorType((1, 1, 2, 2), DType.INT8, Layout.NHWC, QPARAMS),
            "extra": TensorType((1, 1, 2, 1), DType.INT8, Layout.NHWC, QPARAMS),
            "output": TensorType((1, 1, 2, 3), DType.INT8, Layout.NHWC, QPARAMS),
        },
        constants={"extra": np.asarray([[[[50], [60]]]], dtype=np.int8)},
        ops=(
            ReshapeOp("reshape", "input", "image"),
            ConcatenateOp("concat", ("image", "extra"), "output", 3),
        ),
        inputs=("input",),
        outputs=("output",),
    )


def _slice_graph() -> QuantizedGraph:
    return QuantizedGraph(
        name="static_slice",
        values={
            "input": TensorType((1, 3, 5, 2), DType.INT8, Layout.NHWC, QPARAMS),
            "output": TensorType((1, 3, 2, 2), DType.INT8, Layout.NHWC, QPARAMS),
        },
        constants={},
        ops=(SliceOp("slice", "input", "output", axis=2, start=1, stop=5, step=2),),
        inputs=("input",),
        outputs=("output",),
    )


def _run_generated(artifacts, inputs: np.ndarray) -> np.ndarray:  # type: ignore[no-untyped-def]
    compiler = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is required for portable-C differential tests")
    metadata = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    symbol = metadata["model"]
    macro = symbol.upper()
    runner = artifacts.output_dir / "runner.c"
    runner.write_text(
        f'''#include "{artifacts.header.name}"
#include <stdio.h>
#include <string.h>
int main(void) {{
    uint8_t guarded[{macro}_ARENA_SIZE + 32u];
    int8_t input[{macro}_INPUT_SIZE];
    int8_t output[{macro}_OUTPUT_SIZE];
    memset(guarded, 0xA5, sizeof(guarded));
    uint8_t *arena = {macro}_ARENA_SIZE == 0u ? NULL : guarded + 16u;
    while (fread(input, 1u, {macro}_INPUT_SIZE, stdin) == {macro}_INPUT_SIZE) {{
        {symbol}_infer(arena, input, output);
        for (size_t i = 0; i < 16u; ++i) {{
            if (guarded[i] != 0xA5u || guarded[16u + {macro}_ARENA_SIZE + i] != 0xA5u) return 9;
        }}
        if (fwrite(output, 1u, {macro}_OUTPUT_SIZE, stdout) != {macro}_OUTPUT_SIZE) return 2;
    }}
    return ferror(stdin) ? 3 : 0;
}}
''',
        encoding="utf-8",
    )
    executable = artifacts.output_dir / "runner"
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-O1",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all",
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
    result = subprocess.run(executable, input=inputs.tobytes(), check=True, capture_output=True)
    return np.frombuffer(result.stdout, dtype=np.int8)


def test_flatten_is_internal_view_and_concat_nc_hand_golden(tmp_path: Path) -> None:
    plan = lower_to_plan(_flatten_concat_graph())
    assert plan.tensors["flat"].storage is Storage.ALIAS
    assert plan.tensors["flat"].alias_of == "input"
    assert plan.steps[0].aliases[0].value == "flat"
    values = np.asarray([[[[1, 2], [3, 4]]]], dtype=np.int8)
    np.testing.assert_array_equal(
        run_reference(plan, values),
        np.asarray([[1, 2, 3, 4, 91, -92]], dtype=np.int8),
    )
    artifacts = generate_portable_c(plan, tmp_path / "flatten")
    kernel_text = artifacts.kernels_source.read_text(encoding="utf-8")
    assert "flatten_view" not in kernel_text
    assert "concatenate_copy_s8" in kernel_text


@pytest.mark.parametrize("factory", [_flatten_concat_graph, _reshape_concat_graph])
def test_shape_random_python_c_bit_exact_and_no_inplace(tmp_path: Path, factory) -> None:  # type: ignore[no-untyped-def]
    plan = lower_to_plan(factory())
    concat_step = plan.steps[1]
    assert concat_step.aliases == ()
    assert plan.tensors[concat_step.output].storage is Storage.OUTPUT
    rng = np.random.default_rng(77)
    inputs = rng.integers(-128, 128, size=(256, 1, 4), dtype=np.int16).astype(np.int8)
    input_shape = plan.tensors[plan.inputs[0]].tensor_type.shape
    expected = np.concatenate(
        [run_reference(plan, value.reshape(input_shape)).reshape(-1) for value in inputs]
    )
    artifacts = generate_portable_c(plan, tmp_path / plan.name)
    actual = _run_generated(artifacts, inputs)
    np.testing.assert_array_equal(actual, expected)


def test_shape_rejects_malformed_views_concat_and_direct_caller_alias() -> None:
    graph = _flatten_concat_graph()
    bad_values = dict(graph.values)
    bad_values["flat"] = TensorType((1, 5), DType.INT8, Layout.NC, QPARAMS)
    with pytest.raises(GraphValidationError, match="number of elements"):
        verify_graph(
            QuantizedGraph(
                graph.name,
                bad_values,
                graph.constants,
                graph.ops,
                graph.inputs,
                graph.outputs,
            )
        )

    bad_values = dict(graph.values)
    bad_values["suffix"] = TensorType(
        (1, 2), DType.INT8, Layout.NC, PerTensorQParams(0.5, -7)
    )
    with pytest.raises(GraphValidationError, match="identical.*qparams"):
        verify_graph(
            QuantizedGraph(
                graph.name,
                bad_values,
                graph.constants,
                graph.ops,
                graph.inputs,
                graph.outputs,
            )
        )

    direct = QuantizedGraph(
        "direct_view",
        {
            "input": TensorType((1, 4), DType.INT8, Layout.NC, QPARAMS),
            "output": TensorType((1, 1, 2, 2), DType.INT8, Layout.NHWC, QPARAMS),
        },
        {},
        (ReshapeOp("reshape", "input", "output"),),
        ("input",),
        ("output",),
    )
    direct_plan = lower_to_plan(direct)
    assert direct_plan.steps[0].materialize
    assert direct_plan.tensors["output"].storage is Storage.OUTPUT


def test_concatenate_may_read_the_same_edge_more_than_once() -> None:
    graph = QuantizedGraph(
        "duplicate_concat_edge",
        {
            "input": TensorType((1, 2), DType.INT8, Layout.NC, QPARAMS),
            "output": TensorType((1, 4), DType.INT8, Layout.NC, QPARAMS),
        },
        {},
        (ConcatenateOp("concat", ("input", "input"), "output", 1),),
        ("input",),
        ("output",),
    )
    plan = lower_to_plan(graph)
    np.testing.assert_array_equal(
        run_reference(plan, np.asarray([[3, -7]], dtype=np.int8)),
        np.asarray([[3, -7, 3, -7]], dtype=np.int8),
    )


def test_static_slice_hand_golden_random_c_and_malformed_contract(tmp_path: Path) -> None:
    graph = _slice_graph()
    plan = lower_to_plan(graph)
    source = np.arange(30, dtype=np.int8).reshape(1, 3, 5, 2)
    np.testing.assert_array_equal(run_reference(plan, source), source[:, :, 1:5:2, :])

    rng = np.random.default_rng(20260814)
    inputs = rng.integers(-128, 128, size=(256, 3, 5, 2), dtype=np.int16).astype(np.int8)
    expected = np.concatenate(
        [run_reference(plan, value.reshape(1, 3, 5, 2)).reshape(-1) for value in inputs]
    )
    actual = _run_generated(generate_portable_c(plan, tmp_path / "slice"), inputs)
    np.testing.assert_array_equal(actual, expected)

    bad_values = dict(graph.values)
    bad_values["output"] = TensorType((1, 3, 3, 2), DType.INT8, Layout.NHWC, QPARAMS)
    with pytest.raises(GraphValidationError, match="Slice output shape"):
        verify_graph(
            QuantizedGraph(
                graph.name,
                bad_values,
                graph.constants,
                graph.ops,
                graph.inputs,
                graph.outputs,
            )
        )
