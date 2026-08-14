from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from bakenn.backend.portable_c import generate_portable_c
import bakenn.backend.portable_c.families.pool  # noqa: F401 - installs emitter registrations
from bakenn.errors import GraphValidationError
from bakenn.ir import DType, Layout, PerTensorQParams, QuantizedGraph, TensorType, verify_graph
from bakenn.ir.ops.pool import AveragePool2DOp, MaxPool2DOp
import bakenn.ir.verifiers.pool  # noqa: F401 - installs verifier registrations
from bakenn.plan import lower_to_plan
import bakenn.plan.lowering.pool  # noqa: F401 - installs lowering registrations
from bakenn.reference import run_reference
from bakenn.reference.kernels.pool import _round_divide_half_away


def _graph(
    op_type: type[AveragePool2DOp] | type[MaxPool2DOp],
    *,
    input_shape: tuple[int, int, int, int] = (1, 3, 4, 2),
    output_shape: tuple[int, int, int, int] = (1, 2, 4, 2),
    kernel: tuple[int, int] = (2, 2),
    stride: tuple[int, int] = (2, 1),
    padding: tuple[int, int, int, int] = (1, 0, 1, 0),
    input_qparams: PerTensorQParams | None = None,
    output_qparams: PerTensorQParams | None = None,
) -> QuantizedGraph:
    input_qparams = input_qparams or PerTensorQParams(0.125, 3)
    output_qparams = output_qparams or input_qparams
    return QuantizedGraph(
        name=f"test_{op_type.__name__}",
        values={
            "input": TensorType(input_shape, DType.INT8, Layout.NHWC, input_qparams),
            "output": TensorType(output_shape, DType.INT8, Layout.NHWC, output_qparams),
        },
        constants={},
        ops=(op_type("pool", "input", "output", kernel, stride, padding),),
        inputs=("input",),
        outputs=("output",),
    )


def _run_generated(artifacts, inputs: np.ndarray) -> np.ndarray:  # type: ignore[no-untyped-def]
    compiler = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is required for portable-C differential tests")
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    symbol = manifest["model"]
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
    memset(guarded, 0x5A, sizeof(guarded));
    uint8_t *arena = {macro}_ARENA_SIZE == 0u ? NULL : guarded + 16u;
    while (fread(input, 1u, {macro}_INPUT_SIZE, stdin) == {macro}_INPUT_SIZE) {{
        {symbol}_infer(arena, input, output);
        for (size_t i = 0; i < 16u; ++i) {{
            if (guarded[i] != 0x5Au || guarded[16u + {macro}_ARENA_SIZE + i] != 0x5Au) return 9;
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


def test_average_pool_padding_valid_count_and_negative_ties_hand_golden() -> None:
    graph = _graph(
        AveragePool2DOp,
        input_shape=(1, 2, 3, 1),
        output_shape=(1, 2, 2, 1),
        kernel=(2, 2),
        stride=(1, 2),
        padding=(1, 0, 1, 0),
    )
    plan = lower_to_plan(graph)
    values = np.asarray([[[[-2], [-1], [-2]], [[-1], [-2], [-1]]]], dtype=np.int8)
    np.testing.assert_array_equal(
        run_reference(plan, values),
        np.asarray([[[[-2], [-2]], [[-2], [-2]]]], dtype=np.int8),
    )
    assert _round_divide_half_away(-3, 2) == -2
    assert _round_divide_half_away(3, 2) == 2


@pytest.mark.parametrize("op_type", [AveragePool2DOp, MaxPool2DOp])
def test_pool_random_python_c_bit_exact_with_sanitizers(tmp_path: Path, op_type) -> None:  # type: ignore[no-untyped-def]
    plan = lower_to_plan(_graph(op_type))
    artifacts = generate_portable_c(plan, tmp_path / op_type.__name__)
    rng = np.random.default_rng(20260814)
    edge = np.stack(
        [
            np.full((1, 3, 4, 2), -128, dtype=np.int8),
            np.full((1, 3, 4, 2), 127, dtype=np.int8),
        ]
    )
    random = rng.integers(-128, 128, size=(256, 1, 3, 4, 2), dtype=np.int16).astype(np.int8)
    inputs = np.concatenate((edge, random), axis=0)
    expected = np.concatenate([run_reference(plan, value).reshape(-1) for value in inputs])
    actual = _run_generated(artifacts, inputs).reshape(-1)
    np.testing.assert_array_equal(actual, expected)


def test_pool_rejects_mismatched_qparams_bad_shape_empty_window_and_overflow() -> None:
    with pytest.raises(GraphValidationError, match="qparams must be identical"):
        verify_graph(
            _graph(
                AveragePool2DOp,
                output_qparams=PerTensorQParams(0.25, 3),
            )
        )
    with pytest.raises(GraphValidationError, match="output shape"):
        verify_graph(_graph(MaxPool2DOp, output_shape=(1, 2, 2, 2)))
    with pytest.raises(GraphValidationError, match="at least one input"):
        verify_graph(
            _graph(
                MaxPool2DOp,
                input_shape=(1, 1, 1, 1),
                output_shape=(1, 2, 1, 1),
                kernel=(1, 1),
                stride=(1, 1),
                padding=(1, 0, 0, 0),
            )
        )
    with pytest.raises(GraphValidationError, match="exceeds int32"):
        verify_graph(
            _graph(
                AveragePool2DOp,
                input_shape=(1, 1, 1, 1),
                output_shape=(1, 1, 1, 1),
                kernel=(10_000_000, 1),
                stride=(1, 1),
                padding=(0, 9_999_999, 0, 0),
                input_qparams=PerTensorQParams(1.0, 127),
            )
        )
