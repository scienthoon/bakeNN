from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

import bakenn
from bakenn.errors import CompileError, GraphValidationError
from bakenn.ir import (
    DType,
    Layout,
    LinearOp,
    PerAxisQParams,
    PerTensorQParams,
    QuantizedGraph,
    TensorType,
    Conv2DOp,
    verify_graph,
)
from bakenn.plan import lower_to_plan


def one_linear_graph(*, input_scale=2.0, input_zero_point=0, output_scale=0.5):
    values = {
        "input": TensorType((1, 1), DType.INT8, Layout.NC, PerTensorQParams(input_scale, input_zero_point)),
        "weight": TensorType((1, 1), DType.INT8, Layout.OI, PerAxisQParams((1.0,), (0,), 0)),
        "bias": TensorType(
            (1,), DType.INT32, Layout.C, PerAxisQParams((input_scale,), (0,), 0)
        ),
        "output": TensorType((1, 1), DType.INT8, Layout.NC, PerTensorQParams(output_scale, 0)),
    }
    return QuantizedGraph(
        name="positive_shift",
        values=values,
        constants={
            "weight": np.asarray([[1]], dtype=np.int8),
            "bias": np.asarray([0], dtype=np.int32),
        },
        ops=(LinearOp("linear", "input", "weight", "bias", "output"),),
        inputs=("input",),
        outputs=("output",),
    )


class ContractTests(unittest.TestCase):
    def test_quantization_fields_reject_fractional_or_boolean_integers(self):
        with self.assertRaises(ValueError):
            PerTensorQParams(1.0, 0.5)
        with self.assertRaises(ValueError):
            PerTensorQParams(1.0, True)
        with self.assertRaises(ValueError):
            PerAxisQParams((1.0,), (0.5,), 0)
        with self.assertRaises(ValueError):
            PerAxisQParams((1.0,), (128,), 0)
        with self.assertRaisesRegex(ValueError, "outside the tensor rank"):
            TensorType(
                (1,), DType.INT8, Layout.C, PerAxisQParams((1.0,), (0,), 1)
            )
        with self.assertRaisesRegex(ValueError, "qparam count"):
            TensorType(
                (2,), DType.INT8, Layout.C, PerAxisQParams((1.0,), (0,), 0)
            )
        with self.assertRaises(ValueError):
            LinearOp("bad", "x", "w", "b", "y", activation_min=-1.5)

    def test_scale_contract_is_float32_and_rejects_non_deployable_values(self):
        qparams = PerTensorQParams(38198.09059785574, 0)
        self.assertEqual(qparams.scale, float(np.float32(38198.09059785574)))
        for scale in (1e-300, 1e300):
            with self.assertRaises(ValueError):
                PerTensorQParams(scale, 0)

        plan = lower_to_plan(
            one_linear_graph(input_scale=qparams.scale, output_scale=qparams.scale)
        )
        source = np.asarray([[2807559.5]], dtype=np.float32)
        expected = int(np.clip(np.floor(source[0, 0] / np.float32(qparams.scale) + 0.5), -128, 127))
        self.assertEqual(int(bakenn.quantize_input(plan, source)[0, 0]), expected)

    def test_target_size_and_coordinate_fields_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "dimensions"):
            TensorType(
                ((1 << 31),), DType.INT8, Layout.C, PerTensorQParams(1.0, 0)
            )
        with self.assertRaisesRegex(ValueError, "storage"):
            TensorType(
                (65536, 65536), DType.INT8, Layout.NC, PerTensorQParams(1.0, 0)
            )

        q = PerTensorQParams(1.0, 0)
        weight_q = PerAxisQParams((1.0,), (0,), 0)
        graph = QuantizedGraph(
            name="huge_coordinate",
            values={
                "input": TensorType((1, 1, 1, 1), DType.INT8, Layout.NHWC, q),
                "weight": TensorType((1, 1, 1, 1), DType.INT8, Layout.OHWI, weight_q),
                "bias": TensorType((1,), DType.INT32, Layout.C, weight_q),
                "output": TensorType((1, 1, 1, 1), DType.INT8, Layout.NHWC, q),
            },
            constants={
                "weight": np.asarray([[[[1]]]], dtype=np.int8),
                "bias": np.asarray([0], dtype=np.int32),
            },
            ops=(
                Conv2DOp(
                    "conv",
                    "input",
                    "weight",
                    "bias",
                    "output",
                    stride=((1 << 32) + 1, 1),
                    padding=((1 << 32), 0, 0, 0),
                ),
            ),
            inputs=("input",),
            outputs=("output",),
        )
        with self.assertRaisesRegex(GraphValidationError, "32-bit target ABI"):
            verify_graph(graph)

    def test_output_must_be_produced_and_graph_must_be_connected(self):
        graph = one_linear_graph()
        with self.assertRaises(GraphValidationError):
            verify_graph(replace(graph, outputs=("input",)))
        with self.assertRaises(GraphValidationError):
            verify_graph(replace(graph, constants={"bias": graph.constants["bias"]}))
        with self.assertRaisesRegex(GraphValidationError, "unused constants"):
            verify_graph(
                replace(
                    graph,
                    values={
                        **graph.values,
                        "junk": TensorType(
                            (1,), DType.INT8, Layout.C, PerTensorQParams(1.0, 0)
                        ),
                    },
                    constants={
                        **graph.constants,
                        "junk": np.asarray([1], dtype=np.int8),
                    },
                )
            )

    def test_constants_are_immutable_snapshots(self):
        graph = one_linear_graph()
        with self.assertRaises(ValueError):
            graph.constants["weight"][0, 0] = 9
        with self.assertRaises(TypeError):
            graph.constants["new"] = np.asarray([1], dtype=np.int8)  # type: ignore[index]
        plan = lower_to_plan(graph)
        with self.assertRaises(ValueError):
            plan.constants["weight"][0, 0] = 9

    def test_centered_input_rounding(self):
        plan = lower_to_plan(one_linear_graph(input_scale=1.0, input_zero_point=-128, output_scale=1.0))
        actual = bakenn.quantize_input(plan, np.asarray([[0.5]], dtype=np.float32))
        self.assertEqual(int(actual[0, 0]), -127)
        with self.assertRaises(CompileError):
            bakenn.quantize_input(plan, np.asarray([[np.nan]], dtype=np.float32))
        with self.assertRaisesRegex(CompileError, "real numeric dtype"):
            bakenn.quantize_input(plan, np.asarray([[1.0 + 2.0j]], dtype=np.complex64))
        with self.assertRaisesRegex(CompileError, "shape"):
            bakenn.quantize_input(plan, np.asarray([0.5], dtype=np.float32))
        with self.assertRaisesRegex(CompileError, "shape"):
            bakenn.run_reference(plan, np.asarray([0], dtype=np.int8))

    def test_zero_weight_nonzero_bias_uses_explicit_constant_channel_policy(self):
        model = bakenn.FloatMLP(
            (bakenn.FloatLinear(np.zeros((1, 1), dtype=np.float32), np.asarray([0.1], dtype=np.float32)),)
        )
        graph = bakenn.quantize_ptq(model, np.asarray([[-1.0], [1.0]], dtype=np.float32))
        plan = lower_to_plan(graph)
        first = bakenn.run_reference(plan, np.asarray([[-128]], dtype=np.int8))
        second = bakenn.run_reference(plan, np.asarray([[127]], dtype=np.int8))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(
            bakenn.dequantize_output(plan, first),
            np.asarray([[0.1]], dtype=np.float32),
            atol=graph.values[graph.outputs[0]].qparams.scale,
        )
        with self.assertRaisesRegex(CompileError, "at least one sample"):
            bakenn.quantize_ptq(
                bakenn.FloatMLP((bakenn.FloatLinear(np.ones((1, 1), dtype=np.float32)),)),
                np.empty((0, 1), dtype=np.float32),
            )

    def test_ptq_hand_calculated_weight_and_bias(self):
        model = bakenn.FloatMLP(
            (
                bakenn.FloatLinear(
                    np.asarray([[-1.0, 0.0, 1.0]], dtype=np.float32),
                    np.asarray([0.5], dtype=np.float32),
                    name="golden",
                ),
            )
        )
        calibration = np.asarray([[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], dtype=np.float32)
        graph = bakenn.quantize_ptq(model, calibration)
        np.testing.assert_array_equal(
            graph.constants["golden.weight"], np.asarray([[-127, 0, 127]], dtype=np.int8)
        )
        self.assertEqual(graph.values["input"].qparams.zero_point, -1)
        self.assertEqual(int(graph.constants["golden.bias"][0]), 8096)

    def test_generated_c_positive_shift_and_zero_arena(self):
        compiler = os.environ.get("CC", "cc")
        if shutil.which(compiler) is None:
            if os.environ.get("BAKENN_REQUIRE_CC") == "1":
                self.fail(f"required C compiler not found: {compiler}")
            self.skipTest(f"C compiler not found: {compiler}")
        graph = one_linear_graph()
        with tempfile.TemporaryDirectory() as temporary:
            compiled = bakenn.compile(graph, Path(temporary))
            self.assertEqual(compiled.plan.arena_size, 0)
            self.assertEqual(compiled.plan.steps[0].shifts, (3,))
            metadata = json.loads(compiled.artifacts.manifest.read_text(encoding="utf-8"))
            symbol = metadata["model"]
            macro = symbol.upper()
            self.assertTrue(symbol.startswith("bknn_"))
            self.assertEqual(compiled.artifacts.header.name, f"{symbol}.h")
            runner = Path(temporary) / "runner.c"
            runner.write_text(
                f"""#include "{compiled.artifacts.header.name}"
#include <stddef.h>
#include <stdio.h>

int main(void) {{
    int8_t input;
    int8_t output;
    if ({macro}_ARENA_SIZE != 0u) {{ return 4; }}
    while (fread(&input, 1u, 1u, stdin) == 1u) {{
        {symbol}_infer(NULL, &input, &output);
        if (fwrite(&output, 1u, 1u, stdout) != 1u) {{ return 2; }}
    }}
    return ferror(stdin) ? 3 : 0;
}}
""",
                encoding="utf-8",
            )
            executable = Path(temporary) / "runner"
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
                    str(compiled.artifacts.model_source),
                    str(compiled.artifacts.weights_source),
                    str(compiled.artifacts.kernels_source),
                    str(runner),
                    "-I",
                    temporary,
                    "-o",
                    str(executable),
                ],
                check=True,
                capture_output=True,
            )
            inputs = np.asarray([-5, -1, 0, 1, 5], dtype=np.int8)
            result = subprocess.run(executable, input=inputs.tobytes(), capture_output=True, check=True)
            actual = np.frombuffer(result.stdout, dtype=np.int8)
            np.testing.assert_array_equal(actual, np.asarray([-20, -4, 0, 4, 20], dtype=np.int8))
            header = compiled.artifacts.header.read_text(encoding="utf-8")
            self.assertIn(f"#define {macro}_ARENA_SIZE 0u", header)
            self.assertIn(f"#define {macro}_INPUT_SCALE", header)
            self.assertIn(f"#define {macro}_OUTPUT_ZERO_POINT", header)
            self.assertIn(f"#define {macro}_INPUT_LAYOUT BKNN_LAYOUT_NC", header)
            self.assertIn(f"#define {macro}_INPUT_RANK 2u", header)
            self.assertIn(f"#define {macro}_INPUT_DIM_1 1u", header)
            self.assertIn(f"#define {macro}_INPUT_BYTES 1u", header)
            self.assertIn("uint8_t *restrict arena", header)
            self.assertIn("const int8_t *restrict input", header)
            self.assertRegex(header, rf"#define {macro}_INPUT_SCALE 0x[0-9a-f.]+p[+-][0-9]+f")


if __name__ == "__main__":
    unittest.main()
