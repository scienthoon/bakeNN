from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

import bakenn
from bakenn.ir import QuantizedGraph, verify_graph
from tests.p0.model_fixtures import (
    mobilenet_v1_graph,
    representative_graphs,
    residual_ds_cnn_graph,
    tiny_cnn_graph,
)


_FACTORIES = (tiny_cnn_graph, residual_ds_cnn_graph, mobilenet_v1_graph)


def _edge_and_random_inputs(graph: QuantizedGraph, seed: int) -> np.ndarray:
    tensor_type = graph.values[graph.inputs[0]]
    input_shape = tensor_type.shape
    element_shape = input_shape[1:]
    zero_point = tensor_type.qparams.zero_point
    ramp = ((np.arange(tensor_type.numel, dtype=np.int16) * 37) % 256 - 128).astype(
        np.int8
    )
    edge = np.stack(
        (
            np.full(element_shape, -128, dtype=np.int8),
            np.full(element_shape, 127, dtype=np.int8),
            np.full(element_shape, zero_point, dtype=np.int8),
            ramp.reshape(element_shape),
        )
    )
    rng = np.random.default_rng(seed)
    random = rng.integers(-128, 128, size=(32, *element_shape), dtype=np.int16).astype(np.int8)
    return np.concatenate((edge, random), axis=0)


def _compile_runner(compiled, directory: Path, compiler: str) -> Path:  # type: ignore[no-untyped-def]
    artifacts = compiled.artifacts
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    symbol = manifest["model"]
    macro = symbol.upper()
    runner = directory / "runner.c"
    runner.write_text(
        f"""#include "{artifacts.header.name}"
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#define GUARD_SIZE 32u

static int guard_ok(const uint8_t *guard) {{
    for (size_t index = 0; index < GUARD_SIZE; ++index) {{
        if (guard[index] != UINT8_C(0xA5)) {{ return 0; }}
    }}
    return 1;
}}

int main(void) {{
    _Alignas({macro}_ARENA_ALIGNMENT)
        uint8_t arena_storage[GUARD_SIZE + {macro}_ARENA_SIZE + GUARD_SIZE];
    struct {{
        uint8_t before[GUARD_SIZE];
        int8_t data[{macro}_INPUT_SIZE];
        uint8_t after[GUARD_SIZE];
    }} input;
    struct {{
        uint8_t before[GUARD_SIZE];
        int8_t data[{macro}_OUTPUT_SIZE];
        uint8_t after[GUARD_SIZE];
    }} output;
    memset(arena_storage, 0xA5, sizeof(arena_storage));
    memset(&input, 0xA5, sizeof(input));
    memset(&output, 0xA5, sizeof(output));
    uint8_t *arena = {macro}_ARENA_SIZE == 0u ? NULL : arena_storage + GUARD_SIZE;
    while (fread(input.data, 1u, {macro}_INPUT_SIZE, stdin) == {macro}_INPUT_SIZE) {{
        {symbol}_infer(arena, input.data, output.data);
        if (!guard_ok(arena_storage)
            || !guard_ok(arena_storage + GUARD_SIZE + {macro}_ARENA_SIZE)
            || !guard_ok(input.before) || !guard_ok(input.after)
            || !guard_ok(output.before) || !guard_ok(output.after)) {{
            return 4;
        }}
        if (fwrite(output.data, 1u, {macro}_OUTPUT_SIZE, stdout) != {macro}_OUTPUT_SIZE) {{
            return 2;
        }}
    }}
    return ferror(stdin) ? 3 : 0;
}}
""",
        encoding="utf-8",
    )
    executable = directory / "runner"
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
            str(directory),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
    )
    return executable


def test_representative_fixtures_are_deterministic_valid_and_exercise_memory_planning(
    tmp_path: Path,
) -> None:
    graphs = representative_graphs()
    assert [graph.name for graph in graphs] == [
        "p0_tiny_cnn",
        "p0_residual_ds_cnn",
        "p0_mobilenet_v1",
    ]
    assert [[op.name for op in graph.ops] for graph in graphs] == [
        ["conv", "pool", "flatten", "classifier"],
        ["stem", "depthwise", "project", "residual_add"],
        ["depthwise", "pointwise", "global_pool", "flatten", "classifier"],
    ]
    for graph in graphs:
        verify_graph(graph)
        rebuilt = next(item for item in representative_graphs() if item.name == graph.name)
        assert tuple(graph.values) == tuple(rebuilt.values)
        assert tuple(graph.constants) == tuple(rebuilt.constants)
        assert tuple(type(op) for op in graph.ops) == tuple(type(op) for op in rebuilt.ops)
        for name in graph.constants:
            np.testing.assert_array_equal(graph.constants[name], rebuilt.constants[name])

    tiny = bakenn.compile(graphs[0], tmp_path / "tiny").plan
    assert tiny.tensors["flatten.output"].alias_of == "pool.output"
    assert tiny.alias_groups["pool.output"] == ("pool.output", "flatten.output")

    residual = bakenn.compile(graphs[1], tmp_path / "residual").plan
    # block.residual is consumed both by depthwise and by the final Add.  It
    # must remain live while both main-path activation buffers coexist.
    residual_offsets = {
        residual.tensors[name].offset
        for name in ("block.residual", "depthwise.output", "project.output")
    }
    assert len(residual_offsets) == 3

    mobile = bakenn.compile(graphs[2], tmp_path / "mobile").plan
    assert mobile.tensors["flatten.output"].alias_of == "pool.output"
    assert mobile.tensors["pool.output"].offset == mobile.tensors["depthwise.output"].offset


@pytest.mark.parametrize("compiler", ["gcc", "clang"])
@pytest.mark.parametrize("factory", _FACTORIES, ids=("tiny_cnn", "residual_ds_cnn", "mobilenet_v1"))
def test_representative_graph_python_c_bit_exact_strict_sanitized(
    tmp_path: Path,
    factory,
    compiler: str,
) -> None:  # type: ignore[no-untyped-def]
    if shutil.which(compiler) is None:
        pytest.skip(f"{compiler} is not installed")
    graph = factory()
    output_dir = tmp_path / f"{graph.name}_{compiler}"
    compiled = bakenn.compile(graph, output_dir)
    executable = _compile_runner(compiled, output_dir, compiler)
    inputs = _edge_and_random_inputs(graph, seed=20260814)
    input_shape = graph.values[graph.inputs[0]].shape
    expected = np.concatenate(
        [bakenn.run_reference(compiled.plan, sample.reshape(input_shape)) for sample in inputs],
        axis=0,
    )
    result = subprocess.run(executable, input=inputs.tobytes(), capture_output=True, check=True)
    actual = np.frombuffer(result.stdout, dtype=np.int8).reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected)

    manifest = json.loads(compiled.artifacts.manifest.read_text(encoding="utf-8"))
    assert manifest["arithmetic_profile"] == "bakenn.int8.v1"
    assert manifest["arena_bytes"] == compiled.plan.arena_size
    assert manifest["activation_arena_bytes"] == compiled.plan.activation_arena_size
    assert manifest["scratch_bytes"] == compiled.plan.scratch_size
    assert manifest["scratch_offset"] == compiled.plan.scratch_offset
    assert manifest["scratch_alignment"] == compiled.plan.scratch_alignment
    assert manifest["arena_alignment"] == compiled.plan.arena_alignment
    assert manifest["arena_bytes"] >= compiled.plan.activation_arena_size
    assert manifest["constant_bytes"] >= sum(array.nbytes for array in graph.constants.values())
    assert manifest["input"]["shape"] == list(graph.values[graph.inputs[0]].shape)
    assert manifest["output"]["shape"] == list(graph.values[graph.outputs[0]].shape)
    assert manifest["input"]["dtype"] == "int8"
    assert manifest["output"]["dtype"] == "int8"
    assert manifest["input"]["layout"] == graph.values[graph.inputs[0]].layout.value
    assert manifest["output"]["layout"] == graph.values[graph.outputs[0]].layout.value
    assert manifest["input"]["qparams"]["kind"] == "per_tensor"
    assert manifest["output"]["qparams"]["kind"] == "per_tensor"
    assert [operation["kind"] for operation in manifest["operations"]] == [
        step.kernel_kind for step in compiled.plan.steps
    ]
    assert [operation["name"] for operation in manifest["operations"]] == [
        step.name for step in compiled.plan.steps
    ]
    assert all(operation["kind"] for operation in manifest["operations"])

    generated = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            compiled.artifacts.model_source,
            compiled.artifacts.weights_source,
            compiled.artifacts.kernels_source,
        )
    )
    for forbidden in ("malloc(", "calloc(", "realloc(", "free(", "float ", "double "):
        assert forbidden not in generated
