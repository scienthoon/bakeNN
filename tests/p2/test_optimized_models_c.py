from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import numpy as np

import bakenn
from bakenn.ir import AddOp, LinearOp, QuantizedGraph
from tests.p0.model_fixtures import residual_ds_cnn_graph
from tests.p0.test_models_c import (
    _compile_runner,
    _edge_and_random_inputs,
)

from .support import require_compiler
from .test_backend_selection import linear_graph


def _assert_model_differential(
    graph: QuantizedGraph,
    compiled: bakenn.compiler.CompiledModel,
    executable: Path,
    inputs: np.ndarray,
) -> None:
    input_shape = graph.values[graph.inputs[0]].shape
    expected = np.concatenate(
        [
            bakenn.run_reference(compiled.plan, sample.reshape(input_shape))
            for sample in inputs
        ],
        axis=0,
    )
    result = subprocess.run(
        executable, input=inputs.tobytes(), capture_output=True, check=True
    )
    actual = np.frombuffer(result.stdout, dtype=np.int8).reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected)


def test_mixed_optimized_and_portable_whole_model_is_byte_exact(
    tmp_path: Path,
) -> None:
    compiler = require_compiler(os.environ.get("CC", "cc"))
    graph = residual_ds_cnn_graph()
    compiled = bakenn.compile(
        graph,
        tmp_path,
        backend_options=bakenn.CBackendOptions(
            kernel_policy=bakenn.KernelPolicy.AUTO
        ),
    )
    implementations = [
        selection.kernel_id
        for selection in compiled.artifacts.backend_plan.selections
    ]
    assert implementations == [
        "optimized.conv2d_1x1_o2.v1",
        "optimized.depthwise_3x3_c2.v1",
        "optimized.conv2d_1x1_o2.v1",
        "portable.add_s8.v1",
    ]
    executable = _compile_runner(compiled, tmp_path, compiler)
    inputs = _edge_and_random_inputs(graph, seed=20260821)
    _assert_model_differential(graph, compiled, executable, inputs)

    manifest = json.loads(compiled.artifacts.manifest.read_text(encoding="utf-8"))
    assert manifest["backend"]["optimized_steps"] == 3
    assert manifest["activation_arena_bytes"] == compiled.plan.activation_arena_size
    assert manifest["constant_payload_bytes"] == manifest["constant_bytes"]


def _tied_linear_graph() -> QuantizedGraph:
    base = linear_graph(12, 6)
    values = dict(base.values)
    output_type = values.pop("output")
    values.update(left=output_type, right=output_type, output=output_type)
    return QuantizedGraph(
        name="p2_tied_linear",
        values=values,
        constants=base.constants,
        ops=(
            LinearOp("left_linear", "input", "weight", "bias", "left"),
            LinearOp("right_linear", "input", "weight", "bias", "right"),
            AddOp("sum", "left", "right", "output"),
        ),
        inputs=("input",),
        outputs=("output",),
    )


def test_tied_linear_weight_emits_one_shared_packed_constant(
    tmp_path: Path,
) -> None:
    compiler = require_compiler(os.environ.get("CC", "cc"))
    graph = _tied_linear_graph()
    compiled = bakenn.compile(
        graph,
        tmp_path,
        backend_options=bakenn.CBackendOptions(
            kernel_policy=bakenn.KernelPolicy.AUTO
        ),
    )
    backend = compiled.artifacts.backend_plan
    assert [selection.kernel_id for selection in backend.selections] == [
        "optimized.linear_oi2.v1",
        "optimized.linear_oi2.v1",
        "portable.add_s8.v1",
    ]
    assert tuple(backend.packed_constants) == ("weight.linear_oi2",)
    weights_source = compiled.artifacts.weights_source.read_text(encoding="utf-8")
    assert weights_source.count("const int8_t ") == 1

    executable = _compile_runner(compiled, tmp_path, compiler)
    rng = np.random.default_rng(20260822)
    inputs = rng.integers(-128, 128, size=(512, 12), dtype=np.int16).astype(
        np.int8
    )
    inputs[:3] = np.asarray(
        [[-128] * 12, [127] * 12, [-7] * 12], dtype=np.int8
    )
    _assert_model_differential(graph, compiled, executable, inputs)
