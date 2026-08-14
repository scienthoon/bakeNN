from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

# Install all elementwise singledispatch registrations explicitly.  Central
# aggregators are owned by the P0 integration work package.
import bakenn.backend.portable_c.families.elementwise  # noqa: F401
import bakenn.ir.verifiers.elementwise  # noqa: F401
import bakenn.plan.lowering.elementwise  # noqa: F401
import bakenn.reference.kernels.elementwise  # noqa: F401
from bakenn.backend.portable_c.generator import generate_portable_c
from bakenn.ir.graph import QuantizedGraph
from bakenn.ir.ops.elementwise import AddOp, ClampOp, MulOp, RequantizeOp
from bakenn.ir.types import DType, Layout, PerTensorQParams, TensorType
from bakenn.plan.lower import lower_to_plan
from bakenn.reference.executor import run_reference


def _chain_graph(count: int = 17) -> QuantizedGraph:
    shape = (1, count)

    def tensor(scale: float, zero_point: int) -> TensorType:
        return TensorType(
            shape,
            DType.INT8,
            Layout.NC,
            PerTensorQParams(scale, zero_point),
        )

    rng = np.random.default_rng(7301)
    return QuantizedGraph(
        name="elementwise_chain",
        values={
            "input": tensor(0.25, -3),
            "requantized": tensor(0.5, 5),
            "clipped": tensor(0.5, 5),
            "rhs": tensor(0.125, -10),
            "added": tensor(0.25, 2),
            "factor": tensor(0.5, 7),
            "output": tensor(0.2, -5),
        },
        constants={
            "rhs": rng.integers(-128, 128, size=shape, dtype=np.int16).astype(np.int8),
            "factor": rng.integers(-128, 128, size=shape, dtype=np.int16).astype(np.int8),
        },
        ops=(
            RequantizeOp("requantize", "input", "requantized"),
            ClampOp("clamp", "requantized", "clipped", -20, 45),
            AddOp("add", "clipped", "rhs", "added", -100, 100),
            MulOp("mul", "added", "factor", "output", -120, 110),
        ),
        inputs=("input",),
        outputs=("output",),
    )


def _available_compilers() -> list[str]:
    result: list[str] = []
    resolved: set[str] = set()
    for candidate in ("cc", "clang", "gcc"):
        path = shutil.which(candidate)
        if path is None:
            continue
        real = str(Path(path).resolve())
        if real not in resolved:
            resolved.add(real)
            result.append(path)
    return result


def test_elementwise_generated_c_random_differential_strict_and_sanitized(tmp_path: Path) -> None:
    compilers = _available_compilers()
    if not compilers:
        pytest.skip("no C11 compiler is available")

    plan = lower_to_plan(_chain_graph())
    first = generate_portable_c(plan, tmp_path / "first")
    second = generate_portable_c(plan, tmp_path / "second")
    for first_path in sorted(first.output_dir.iterdir()):
        second_path = second.output_dir / first_path.name
        assert second_path.read_bytes() == first_path.read_bytes()

    rng = np.random.default_rng(20260814)
    edge_inputs = np.asarray(
        [
            [-128] * 17,
            [127] * 17,
            [0] * 17,
            [-128, 127, -1, 0, 1, -64, 63, 126, -127, 2, -2, 80, -80, 12, -12, 99, -99],
        ],
        dtype=np.int8,
    )
    random_inputs = rng.integers(-128, 128, size=(256, 17), dtype=np.int16).astype(np.int8)
    inputs = np.concatenate((edge_inputs, random_inputs), axis=0)
    expected = np.concatenate(
        [run_reference(plan, sample.reshape(1, -1)) for sample in inputs], axis=0
    )

    metadata = json.loads(first.manifest.read_text(encoding="utf-8"))
    symbol = metadata["model"]
    macro = symbol.upper()
    runner = first.output_dir / "runner.c"
    runner.write_text(
        f"""#include "{first.header.name}"
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#define GUARD_SIZE 16u

static int guard_ok(const uint8_t *guard) {{
    for (size_t index = 0; index < GUARD_SIZE; ++index) {{
        if (guard[index] != UINT8_C(0xA5)) {{
            return 0;
        }}
    }}
    return 1;
}}

int main(void) {{
    _Alignas({macro}_ARENA_ALIGNMENT)
        uint8_t arena_storage[GUARD_SIZE + {macro}_ARENA_SIZE + GUARD_SIZE];
    struct {{ uint8_t before[GUARD_SIZE]; int8_t data[{macro}_INPUT_SIZE]; uint8_t after[GUARD_SIZE]; }} input;
    struct {{ uint8_t before[GUARD_SIZE]; int8_t data[{macro}_OUTPUT_SIZE]; uint8_t after[GUARD_SIZE]; }} output;
    memset(arena_storage, 0xA5, sizeof(arena_storage));
    memset(&input, 0xA5, sizeof(input));
    memset(&output, 0xA5, sizeof(output));
    uint8_t *arena = arena_storage + GUARD_SIZE;
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
    sources = [first.model_source, first.weights_source, first.kernels_source, runner]
    for compiler_index, compiler in enumerate(compilers):
        for optimization in ("-O0", "-O2"):
            executable = first.output_dir / f"runner_{compiler_index}_{optimization[2:]}"
            subprocess.run(
                [
                    compiler,
                    "-std=c11",
                    optimization,
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-pedantic",
                    "-fsanitize=address,undefined",
                    "-fno-sanitize-recover=all",
                    *(str(path) for path in sources),
                    "-I",
                    str(first.output_dir),
                    "-o",
                    str(executable),
                ],
                check=True,
                capture_output=True,
            )
            completed = subprocess.run(
                executable,
                input=inputs.tobytes(),
                capture_output=True,
                check=True,
            )
            actual = np.frombuffer(completed.stdout, dtype=np.int8).reshape(expected.shape)
            np.testing.assert_array_equal(actual, expected)

    generated = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (first.model_source, first.weights_source, first.kernels_source)
    )
    for forbidden in ("malloc(", "calloc(", "realloc(", "free("):
        assert forbidden not in generated
