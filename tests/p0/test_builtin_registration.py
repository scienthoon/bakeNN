from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_public_import_installs_every_builtin_stage_without_side_effect_imports(
    tmp_path: Path,
) -> None:
    """A clean process must not need family-specific registration imports."""

    script = r'''
from pathlib import Path
import numpy as np
import bakenn
from bakenn.ir import (
    AddOp, DType, Layout, PerTensorQParams, QuantizedGraph, TensorType, verify_graph,
)
from bakenn.plan import lower_to_plan
from bakenn.reference import run_reference

q = PerTensorQParams(0.25, -3)
graph = QuantizedGraph(
    name="public_add",
    values={
        "input": TensorType((1, 4), DType.INT8, Layout.NC, q),
        "rhs": TensorType((1, 4), DType.INT8, Layout.NC, q),
        "output": TensorType((1, 4), DType.INT8, Layout.NC, q),
    },
    constants={"rhs": np.asarray([[1, 2, 3, 4]], dtype=np.int8)},
    ops=(AddOp("add", "input", "rhs", "output"),),
    inputs=("input",),
    outputs=("output",),
)
verify_graph(graph)
plan = lower_to_plan(graph)
actual = run_reference(plan, np.asarray([[5, 6, 7, 8]], dtype=np.int8))
np.testing.assert_array_equal(actual, np.asarray([[9, 11, 13, 15]], dtype=np.int8))
compiled = bakenn.compile(graph, Path(__import__("sys").argv[1]))
assert compiled.artifacts.header.exists()
assert compiled.artifacts.kernels_source.exists()
'''
    environment = dict(os.environ)
    source = str(Path(__file__).resolve().parents[2] / "src")
    environment["PYTHONPATH"] = source
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
