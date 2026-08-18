from __future__ import annotations

import pytest


pytest.importorskip("flatbuffers")
pytest.importorskip("tflite")

from bakenn.errors import CompileError  # noqa: E402
from bakenn.ir import PerAxisQParams  # noqa: E402
from benchmarks.tflm_compare.quantized_graph_to_tflite import (  # noqa: E402
    _collapse_repeated_per_axis,
)


def test_repeated_linear_qparams_collapse_to_tflite_scalar() -> None:
    qparams = PerAxisQParams((0.25, 0.25, 0.25), (0, 0, 0), axis=0)
    collapsed = _collapse_repeated_per_axis(qparams)

    assert collapsed.scale == 0.25
    assert collapsed.zero_point == 0


def test_true_per_channel_linear_qparams_are_not_silently_changed() -> None:
    qparams = PerAxisQParams((0.25, 0.5), (0, 0), axis=0)

    with pytest.raises(CompileError, match="requires per-tensor weight scale"):
        _collapse_repeated_per_axis(qparams)

