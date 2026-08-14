import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

import bakenn


class GeneratedCEndToEndTests(unittest.TestCase):
    def test_generated_c_is_bit_exact_and_deterministic(self):
        compiler = os.environ.get("CC", "cc")
        if shutil.which(compiler) is None:
            if os.environ.get("BAKENN_REQUIRE_CC") == "1":
                self.fail(f"required C compiler not found: {compiler}")
            self.skipTest(f"C compiler not found: {compiler}")

        rng = np.random.default_rng(20260814)
        model = bakenn.FloatMLP(
            (
                bakenn.FloatLinear(rng.normal(0, 0.7, (7, 8)), rng.normal(0, 0.2, 7), True, "hidden_0"),
                bakenn.FloatLinear(rng.normal(0, 0.6, (5, 7)), rng.normal(0, 0.2, 5), True, "hidden_1"),
                bakenn.FloatLinear(rng.normal(0, 0.5, (3, 5)), rng.normal(0, 0.2, 3), False, "logits"),
            ),
            "tiny_mlp",
        )
        calibration = rng.normal(0, 1.4, (256, 8))
        graph = bakenn.quantize_ptq(model, calibration)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = bakenn.compile(graph, root / "first")
            second = bakenn.compile(graph, root / "second")

            first_files = sorted(path.name for path in first.artifacts.output_dir.iterdir())
            second_files = sorted(path.name for path in second.artifacts.output_dir.iterdir())
            self.assertEqual(first_files, second_files)
            for filename in first_files:
                self.assertEqual(
                    (first.artifacts.output_dir / filename).read_bytes(),
                    (second.artifacts.output_dir / filename).read_bytes(),
                )

            edge_inputs = np.asarray(
                [
                    [-128] * 8,
                    [127] * 8,
                    [0] * 8,
                    [-128, 127, -1, 0, 1, -64, 63, 126],
                ],
                dtype=np.int8,
            )
            random_inputs = rng.integers(-128, 128, size=(128, 8), dtype=np.int16).astype(np.int8)
            inputs = np.concatenate((edge_inputs, random_inputs), axis=0)
            expected = np.concatenate(
                [
                    bakenn.run_reference(first.plan, sample.reshape(1, -1)).reshape(1, -1)
                    for sample in inputs
                ],
                axis=0,
            )

            manifest_data = json.loads(first.artifacts.manifest.read_text(encoding="utf-8"))
            c_symbol = manifest_data["model"]
            macro = c_symbol.upper()
            runner = first.artifacts.output_dir / "runner.c"
            runner.write_text(
                f"""#include "{first.artifacts.header.name}"
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
    uint8_t *arena = {macro}_ARENA_SIZE == 0u ? NULL : arena_storage + GUARD_SIZE;
    while (fread(input.data, sizeof(input.data[0]), {macro}_INPUT_SIZE, stdin) == {macro}_INPUT_SIZE) {{
        {c_symbol}_infer(arena, input.data, output.data);
        if (!guard_ok(arena_storage) || !guard_ok(arena_storage + GUARD_SIZE + {macro}_ARENA_SIZE)
            || !guard_ok(input.before) || !guard_ok(input.after)
            || !guard_ok(output.before) || !guard_ok(output.after)) {{
            return 4;
        }}
        if (fwrite(output.data, sizeof(output.data[0]), {macro}_OUTPUT_SIZE, stdout) != {macro}_OUTPUT_SIZE) {{
            return 2;
        }}
    }}
    return ferror(stdin) ? 3 : 0;
}}
""",
                encoding="utf-8",
            )
            executable = first.artifacts.output_dir / "runner"
            sources = [
                first.artifacts.model_source,
                first.artifacts.weights_source,
                first.artifacts.kernels_source,
                runner,
            ]
            subprocess.run(
                [
                    compiler,
                    "-std=c11",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-pedantic",
                    "-fsanitize=address,undefined",
                    "-fno-sanitize-recover=all",
                    *(str(path) for path in sources),
                    "-I",
                    str(first.artifacts.output_dir),
                    "-o",
                    str(executable),
                ],
                check=True,
                capture_output=True,
            )
            result = subprocess.run(executable, input=inputs.tobytes(), capture_output=True, check=True)
            actual = np.frombuffer(result.stdout, dtype=np.int8).reshape(expected.shape)
            np.testing.assert_array_equal(actual, expected)

            manifest = json.loads(first.artifacts.manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["arena_bytes"], first.plan.arena_size)
            self.assertEqual(manifest["arithmetic_profile"], "bakenn.int8.v1")
            self.assertEqual(manifest["compiler_version"], bakenn.__version__)
            generated_text = "\n".join(path.read_text(encoding="utf-8") for path in sources[:-1])
            for forbidden in ("malloc(", "calloc(", "realloc(", "free("):
                self.assertNotIn(forbidden, generated_text)


if __name__ == "__main__":
    unittest.main()
