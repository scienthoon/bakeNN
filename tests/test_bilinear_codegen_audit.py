"""Independent singleton-axis resize goldens for reference and sanitized C."""

from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

import bakenn
from bakenn.ir import (
    DType, Layout, PerTensorQParams, QuantizedGraph, ResizeBilinear2DOp, TensorType,
)
from bakenn.plan import lower_to_plan
from bakenn.reference import run_reference
from tests.p0.test_models_c import _compile_runner


# With corners aligned, a singleton axis chooses its first input coordinate.
# Without corner alignment, that axis samples the input center. These literal
# goldens avoid relying on the same map for both the implementation and oracle.
_CASES = [
    (True, (1, 1), [1]),
    (True, (1, 3), [1, 2, 3]),
    (True, (3, 1), [1, 4, 7]),
    (False, (1, 1), [5]),
    (False, (1, 3), [4, 5, 6]),
    (False, (3, 1), [2, 5, 8]),
]


def _graph(align_corners: bool, output_size: tuple[int, int]) -> QuantizedGraph:
    qparams = PerTensorQParams(1.0, 0)
    return QuantizedGraph(
        "bilinear_singleton_audit",
        {
            "input": TensorType((1, 3, 3, 1), DType.INT8, Layout.NHWC, qparams),
            "output": TensorType((1, *output_size, 1), DType.INT8, Layout.NHWC, qparams),
        },
        {},
        (ResizeBilinear2DOp("resize", "input", "output", align_corners),),
        ("input",),
        ("output",),
    )


@pytest.mark.parametrize("align_corners,output_size,expected", _CASES)
def test_singleton_bilinear_reference_hand_goldens(
    align_corners: bool, output_size: tuple[int, int], expected: list[int],
) -> None:
    plan = lower_to_plan(_graph(align_corners, output_size))
    values = np.arange(1, 10, dtype=np.int8).reshape(1, 3, 3, 1)
    actual = run_reference(plan, values)
    np.testing.assert_array_equal(actual.reshape(-1), np.asarray(expected, dtype=np.int8))


@pytest.mark.parametrize("align_corners,output_size,expected", _CASES)
def test_singleton_bilinear_sanitized_c_hand_goldens(
    align_corners: bool, output_size: tuple[int, int], expected: list[int], tmp_path: Path,
) -> None:
    compiler = shutil.which("clang") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("host C compiler is unavailable")
    compiled = bakenn.compile(_graph(align_corners, output_size), tmp_path / "generated")
    executable = _compile_runner(compiled, compiled.artifacts.output_dir, compiler)
    values = np.arange(1, 10, dtype=np.int8).reshape(1, 3, 3, 1)
    process = subprocess.run([str(executable)], input=values.tobytes(), capture_output=True, check=True)
    actual = np.frombuffer(process.stdout, dtype=np.int8)
    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.int8))
