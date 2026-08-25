from __future__ import annotations

from dataclasses import replace

import pytest


pytest.importorskip("flatbuffers")
pytest.importorskip("tflite")

from bakenn.errors import CompileError  # noqa: E402
from bakenn.ir import AveragePool2DOp, PerAxisQParams  # noqa: E402
from bakenn.ir.ops.pool import AVERAGE_POOL_PROFILE_TFLITE_RAW_V1  # noqa: E402
from benchmarks.tflm_compare.quantized_graph_to_tflite import (  # noqa: E402
    _collapse_repeated_per_axis,
    export_quantized_graph,
)
from tests.p0.model_fixtures import mobilenet_v1_graph  # noqa: E402


def test_repeated_linear_qparams_collapse_to_tflite_scalar() -> None:
    qparams = PerAxisQParams((0.25, 0.25, 0.25), (0, 0, 0), axis=0)
    collapsed = _collapse_repeated_per_axis(qparams)

    assert collapsed.scale == 0.25
    assert collapsed.zero_point == 0


def test_true_per_channel_linear_qparams_are_not_silently_changed() -> None:
    qparams = PerAxisQParams((0.25, 0.5), (0, 0), axis=0)

    with pytest.raises(CompileError, match="requires per-tensor weight scale"):
        _collapse_repeated_per_axis(qparams)


def test_average_pool_export_rejects_silent_rounding_profile_change() -> None:
    full_graph = mobilenet_v1_graph()
    ops = full_graph.ops[:3]
    required = set(full_graph.inputs)
    for op in ops:
        required.update(op.inputs)
        required.update(op.outputs)
    graph = replace(
        full_graph,
        values={name: value for name, value in full_graph.values.items() if name in required},
        constants={
            name: value for name, value in full_graph.constants.items() if name in required
        },
        ops=ops,
        outputs=ops[-1].outputs,
    )
    with pytest.raises(CompileError, match="refusing to silently change centered"):
        export_quantized_graph(graph)

    tflite_graph = replace(
        graph,
        ops=tuple(
            replace(op, arithmetic_profile=AVERAGE_POOL_PROFILE_TFLITE_RAW_V1)
            if isinstance(op, AveragePool2DOp)
            else op
            for op in graph.ops
        ),
    )
    exported = export_quantized_graph(tflite_graph)
    assert exported.operator_counts["AVERAGE_POOL_2D"] == 1
