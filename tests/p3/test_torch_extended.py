from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

import bakenn
from bakenn.ir import (
    AveragePool1DOp,
    Conv1DOp,
    Conv2DOp,
    HardSwishOp,
    Pad2DOp,
    ReduceMeanOp,
    SigmoidOp,
    SiLUOp,
)
from tests.p0.test_models_c import _compile_runner


torch = pytest.importorskip("torch")
nn = torch.nn
functional = torch.nn.functional


class VisionTrack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(4, 6, 3, padding=1, groups=2, bias=True)

    def forward(self, value):  # type: ignore[no-untyped-def]
        value = functional.hardswish(self.conv(value))
        value = functional.pad(value, (1, 2, 1, 0), mode="constant", value=0.0)
        value = value.mean((2, 3), keepdim=True)
        return torch.sigmoid(value)


class AudioTrack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(4, 6, 3, padding=1, groups=2, bias=False)
        self.bn = nn.BatchNorm1d(6)

    def forward(self, value):  # type: ignore[no-untyped-def]
        value = functional.silu(self.bn(self.conv(value)))
        value = functional.avg_pool1d(
            value,
            kernel_size=2,
            stride=2,
            count_include_pad=False,
        )
        return value.mean((2,), keepdim=True)


@pytest.mark.parametrize(
    ("model", "shape", "expected_types"),
    (
        (
            VisionTrack(),
            (1, 4, 8, 8),
            (Conv2DOp, HardSwishOp, Pad2DOp, ReduceMeanOp, SigmoidOp),
        ),
        (
            AudioTrack(),
            (1, 4, 16),
            (Conv1DOp, SiLUOp, AveragePool1DOp, ReduceMeanOp),
        ),
    ),
)
def test_real_torch_track_captures_ptqs_and_emits_byte_exact_c(
    model: nn.Module,
    shape: tuple[int, ...],
    expected_types: tuple[type[object], ...],
    tmp_path: Path,
) -> None:
    torch.manual_seed(11)
    model.eval()
    example = torch.randn(*shape)
    calibration = torch.randn(8, *shape[1:])
    compiled = bakenn.compile_torch_ptq(
        model,
        example,
        calibration,
        tmp_path / type(model).__name__,
        name=type(model).__name__.lower(),
    )
    operation_types = tuple(type(op) for op in compiled.graph.ops)
    for expected in expected_types:
        assert expected in operation_types
    if isinstance(model, VisionTrack):
        grouped = next(op for op in compiled.graph.ops if isinstance(op, Conv2DOp))
        assert grouped.groups == 2
    else:
        grouped = next(op for op in compiled.graph.ops if isinstance(op, Conv1DOp))
        assert grouped.groups == 2
        assert all("batch_norm" not in type(op).__name__.lower() for op in compiled.float_graph.ops)

    compiler = shutil.which("clang") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("host C compiler is unavailable")
    executable = _compile_runner(compiled, compiled.artifacts.output_dir, compiler)
    input_type = compiled.graph.values[compiled.graph.inputs[0]]
    rng = np.random.default_rng(27)
    samples = rng.integers(
        -128,
        128,
        size=(32, *input_type.shape[1:]),
        dtype=np.int16,
    ).astype(np.int8)
    expected = np.concatenate(
        [
            bakenn.run_reference(compiled.plan, sample.reshape(input_type.shape)).reshape(-1)
            for sample in samples
        ]
    )
    process = subprocess.run(executable, input=samples.tobytes(), capture_output=True, check=True)
    np.testing.assert_array_equal(np.frombuffer(process.stdout, dtype=np.int8), expected)


def test_squeeze_unsqueeze_public_view_materializes_output(tmp_path: Path) -> None:
    class Views(nn.Module):
        def forward(self, value):  # type: ignore[no-untyped-def]
            return value.squeeze(2).unsqueeze(2)

    model = Views().eval()
    compiled = bakenn.compile_torch_ptq(
        model,
        torch.randn(1, 4, 1, 5),
        torch.randn(4, 4, 1, 5),
        tmp_path,
        name="views",
    )
    assert len(compiled.plan.steps) == 2
    assert not compiled.plan.steps[0].materialize
    assert compiled.plan.steps[1].materialize
    model_source = compiled.artifacts.model_source.read_text(encoding="utf-8")
    assert "view_copy_s8(input, output" in model_source
