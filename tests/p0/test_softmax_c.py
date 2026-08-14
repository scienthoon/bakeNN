from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from bakenn.backend.portable_c import generate_portable_c
import bakenn.backend.portable_c.families.softmax  # noqa: F401 - installs emitter registrations
from bakenn.errors import GraphValidationError
from bakenn.ir import DType, Layout, PerTensorQParams, QuantizedGraph, TensorType, verify_graph
from bakenn.ir.ops.softmax import SoftmaxOp
from bakenn.ir.verifiers.softmax import Q15_ONE
import bakenn.ir.verifiers.softmax  # noqa: F401 - installs verifier registrations
from bakenn.plan import lower_to_plan
from bakenn.plan.lowering.softmax import build_softmax_lut
import bakenn.plan.lowering.softmax  # noqa: F401 - installs lowering registrations
from bakenn.reference import run_reference
import bakenn.reference.kernels.softmax  # noqa: F401 - installs reference registrations


OUTPUT_QPARAMS = PerTensorQParams(1.0 / 256.0, -128)


def _graph(class_count: int, *, input_scale: float = math.log(2.0)) -> QuantizedGraph:
    return QuantizedGraph(
        name=f"softmax_{class_count}",
        values={
            "input": TensorType(
                (1, class_count), DType.INT8, Layout.NC, PerTensorQParams(input_scale, 11)
            ),
            "output": TensorType((1, class_count), DType.INT8, Layout.NC, OUTPUT_QPARAMS),
        },
        constants={},
        ops=(SoftmaxOp("softmax", "input", "output"),),
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
    memset(guarded, 0xC3, sizeof(guarded));
    uint8_t *arena = {macro}_ARENA_SIZE == 0u ? NULL : guarded + 16u;
    while (fread(input, 1u, {macro}_INPUT_SIZE, stdin) == {macro}_INPUT_SIZE) {{
        {symbol}_infer(arena, input, output);
        for (size_t i = 0; i < 16u; ++i) {{
            if (guarded[i] != 0xC3u || guarded[16u + {macro}_ARENA_SIZE + i] != 0xC3u) return 9;
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


def test_softmax_lut_and_hand_goldens_are_exact() -> None:
    lut = build_softmax_lut(math.log(2.0))
    assert len(lut) == 256
    assert lut[0] == Q15_ONE
    assert lut[1] == 16384
    assert all(left >= right for left, right in zip(lut, lut[1:]))

    equal_plan = lower_to_plan(_graph(4))
    equal = run_reference(equal_plan, np.asarray([[7, 7, 7, 7]], dtype=np.int8))
    np.testing.assert_array_equal(equal, np.asarray([[-64, -64, -64, -64]], dtype=np.int8))

    dominant_plan = lower_to_plan(_graph(2, input_scale=1.0))
    dominant = run_reference(dominant_plan, np.asarray([[127, -128]], dtype=np.int8))
    np.testing.assert_array_equal(dominant, np.asarray([[127, -128]], dtype=np.int8))


def test_softmax_class_two_exhaustive_python_c_bit_exact(tmp_path: Path) -> None:
    plan = lower_to_plan(_graph(2, input_scale=0.03125))
    left = np.repeat(np.arange(-128, 128, dtype=np.int16), 256)
    right = np.tile(np.arange(-128, 128, dtype=np.int16), 256)
    inputs = np.stack((left, right), axis=1).astype(np.int8).reshape(-1, 1, 2)
    expected = np.concatenate([run_reference(plan, value).reshape(-1) for value in inputs])
    artifacts = generate_portable_c(plan, tmp_path / "exhaustive")
    actual = _run_generated(artifacts, inputs)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("class_count", [3, 10, 100])
def test_softmax_random_python_c_bit_exact_and_deterministic(
    tmp_path: Path, class_count: int
) -> None:
    plan = lower_to_plan(_graph(class_count, input_scale=0.0625))
    rng = np.random.default_rng(9000 + class_count)
    random = rng.integers(
        -128, 128, size=(128, 1, class_count), dtype=np.int16
    ).astype(np.int8)
    equal = np.full((1, 1, class_count), 23, dtype=np.int8)
    inputs = np.concatenate((equal, random), axis=0)
    expected = np.concatenate([run_reference(plan, value).reshape(-1) for value in inputs])
    first = generate_portable_c(plan, tmp_path / "first")
    second = generate_portable_c(plan, tmp_path / "second")
    assert first.weights_source.read_bytes() == second.weights_source.read_bytes()
    assert first.kernels_source.read_bytes() == second.kernels_source.read_bytes()
    manifest = json.loads(first.manifest.read_text(encoding="utf-8"))
    operation = manifest["operations"][0]
    assert operation["arithmetic_profile"] == "bakenn.softmax_lut.q15.v1"
    assert operation["lut_entries"] == 256
    actual = _run_generated(first, inputs)
    np.testing.assert_array_equal(actual, expected)
    codes = actual.astype(np.int16) + 128
    assert np.all((0 <= codes) & (codes <= 255))
    equal_codes = codes[:class_count]
    assert int(equal_codes.max()) - int(equal_codes.min()) <= 1


def test_softmax_rejects_wrong_output_qparams_rank_and_uint32_sum_overflow() -> None:
    graph = _graph(3)
    wrong_values = dict(graph.values)
    wrong_values["output"] = TensorType(
        (1, 3), DType.INT8, Layout.NC, PerTensorQParams(1.0 / 255.0, -128)
    )
    with pytest.raises(GraphValidationError, match="output qparams"):
        verify_graph(
            QuantizedGraph(
                graph.name,
                wrong_values,
                graph.constants,
                graph.ops,
                graph.inputs,
                graph.outputs,
            )
        )

    rank_values = {
        "input": TensorType((1, 1, 1, 3), DType.INT8, Layout.NHWC, PerTensorQParams(0.1, 0)),
        "output": TensorType((1, 1, 1, 3), DType.INT8, Layout.NHWC, OUTPUT_QPARAMS),
    }
    with pytest.raises(GraphValidationError, match="NC layout|rank-two"):
        verify_graph(
            QuantizedGraph(
                "rank4",
                rank_values,
                {},
                (SoftmaxOp("softmax", "input", "output"),),
                ("input",),
                ("output",),
            )
        )

    with pytest.raises(GraphValidationError, match="overflow uint32"):
        verify_graph(_graph(200_000))
