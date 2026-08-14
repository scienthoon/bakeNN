from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

import bakenn
from tests.p0.test_models_c import _compile_runner


torch = pytest.importorskip("torch")
nn = torch.nn
functional = torch.nn.functional


class BottleneckClassifier(nn.Module):
    """ResNet bottleneck topology: 1x1 -> 3x3 -> 1x1 plus skip Add."""

    def __init__(self) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(4, 2, 1)
        self.spatial = nn.Conv2d(2, 2, 3, padding=1)
        self.expand = nn.Conv2d(2, 4, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(4, 3)

    def forward(self, value):  # type: ignore[no-untyped-def]
        residual = value
        value = functional.relu(self.reduce(value))
        value = functional.relu(self.spatial(value))
        value = functional.relu(self.expand(value) + residual)
        return self.classifier(torch.flatten(self.pool(value), 1))


class DenseConcatClassifier(nn.Module):
    """DenseNet-style growing feature list, including torch.cat([x])."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 3, padding=1)
        self.grow1 = nn.Conv2d(4, 2, 3, padding=1)
        self.grow2 = nn.Conv2d(6, 2, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(8, 3)

    def forward(self, value):  # type: ignore[no-untyped-def]
        features = [functional.relu(self.stem(value))]
        first_input = torch.cat(features, dim=1)
        features.append(functional.relu(self.grow1(first_input)))
        features.append(functional.relu(self.grow2(torch.cat(features, dim=1))))
        value = torch.cat(features, dim=1)
        return self.classifier(torch.flatten(self.pool(value), 1))


class InceptionClassifier(nn.Module):
    """Three parallel branches with pointwise, spatial and pooling paths."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 1)
        self.branch1 = nn.Conv2d(4, 2, 1)
        self.branch3 = nn.Conv2d(4, 2, 3, padding=1)
        self.pool_project = nn.Conv2d(4, 2, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(6, 3)

    def forward(self, value):  # type: ignore[no-untyped-def]
        value = functional.relu(self.stem(value))
        branch1 = functional.relu(self.branch1(value))
        branch3 = functional.relu(self.branch3(value))
        pooled = functional.avg_pool2d(
            value, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        pooled = functional.relu(self.pool_project(pooled))
        value = torch.cat((branch1, branch3, pooled), dim=1)
        return self.classifier(torch.flatten(self.pool(value), 1))


class FireClassifier(nn.Module):
    """SqueezeNet Fire topology with parallel 1x1 and 3x3 expand paths."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 3, padding=1)
        self.squeeze = nn.Conv2d(4, 2, 1)
        self.expand1 = nn.Conv2d(2, 3, 1)
        self.expand3 = nn.Conv2d(2, 3, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(6, 3)

    def forward(self, value):  # type: ignore[no-untyped-def]
        value = functional.relu(self.stem(value))
        squeezed = functional.relu(self.squeeze(value))
        value = torch.cat(
            (
                functional.relu(self.expand1(squeezed)),
                functional.relu(self.expand3(squeezed)),
            ),
            dim=1,
        )
        return self.classifier(torch.flatten(self.pool(value), 1))


class AudioFlattenClassifier(nn.Module):
    """Conv1D feature extractor whose non-singleton NCL tensor feeds Linear."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(3, 4, 3, padding=1)
        self.grouped = nn.Conv1d(4, 4, 3, padding=1, groups=2)
        self.classifier = nn.Linear(4 * 6, 3)

    def forward(self, value):  # type: ignore[no-untyped-def]
        value = functional.relu(self.conv(value))
        value = functional.max_pool1d(value, 2, 2)
        value = functional.relu(self.grouped(value))
        return self.classifier(torch.flatten(value, 1))


class SoftmaxMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(12, 9)
        self.output = nn.Linear(9, 5)

    def forward(self, value):  # type: ignore[no-untyped-def]
        return torch.softmax(self.output(functional.relu(self.hidden(value))), dim=-1)


class TemporalMeanClassifier(nn.Module):
    """Residual temporal Conv1D followed by keepdim=False global mean."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(4, 4, 3, padding=1)
        self.conv2 = nn.Conv1d(4, 4, 3, padding=1, groups=2)
        self.classifier = nn.Linear(4, 3)

    def forward(self, value):  # type: ignore[no-untyped-def]
        residual = value
        value = functional.relu(self.conv1(value))
        value = functional.relu(self.conv2(value) + residual)
        return self.classifier(value.mean(dim=2, keepdim=False))


@dataclass(frozen=True)
class ModelCase:
    name: str
    model_type: type[nn.Module]
    input_shape: tuple[int, ...]
    required_float_ops: tuple[str, ...]
    fp_tolerance: float


CASES = (
    ModelCase(
        "zoo_bottleneck",
        BottleneckClassifier,
        (1, 4, 8, 8),
        ("FloatConv2DOp", "FloatAddOp", "FloatLinearOp"),
        0.02,
    ),
    ModelCase(
        "zoo_dense_concat",
        DenseConcatClassifier,
        (1, 3, 8, 8),
        ("FloatConcatOp", "FloatConv2DOp", "FloatLinearOp"),
        0.02,
    ),
    ModelCase(
        "zoo_inception",
        InceptionClassifier,
        (1, 3, 8, 8),
        ("FloatConcatOp", "FloatAveragePool2DOp", "FloatLinearOp"),
        0.02,
    ),
    ModelCase(
        "zoo_fire",
        FireClassifier,
        (1, 3, 8, 8),
        ("FloatConcatOp", "FloatConv2DOp", "FloatLinearOp"),
        0.02,
    ),
    ModelCase(
        "zoo_audio_flatten",
        AudioFlattenClassifier,
        (1, 3, 12),
        ("FloatConv1DOp", "FloatMaxPool1DOp", "FloatLinearOp"),
        0.02,
    ),
    ModelCase(
        "zoo_softmax_mlp",
        SoftmaxMLP,
        (1, 12),
        ("FloatLinearOp", "FloatSoftmaxOp"),
        0.04,
    ),
    ModelCase(
        "zoo_temporal_mean",
        TemporalMeanClassifier,
        (1, 4, 12),
        ("FloatConv1DOp", "FloatAddOp", "FloatReduceMeanOp", "FloatLinearOp"),
        0.02,
    ),
)


def _canonical_input(value: np.ndarray) -> np.ndarray:
    if value.ndim == 4:
        return np.ascontiguousarray(value.transpose(0, 2, 3, 1))
    if value.ndim == 3:
        return np.ascontiguousarray(value.transpose(0, 2, 1))
    return np.ascontiguousarray(value)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_diverse_model_zoo_ptq_reference_and_generated_c(
    case: ModelCase,
    tmp_path: Path,
) -> None:
    seed = 5200 + CASES.index(case)
    torch.manual_seed(seed)
    model = case.model_type().eval()
    calibration = torch.randn(10, *case.input_shape[1:])
    example = calibration[:1]
    compiled = bakenn.compile_torch_ptq(
        model,
        example,
        calibration,
        tmp_path / case.name,
        name=case.name,
    )

    captured = {type(op).__name__ for op in compiled.float_graph.ops}
    assert set(case.required_float_ops) <= captured
    assert compiled.artifacts.model_source.stat().st_size > 0
    assert compiled.artifacts.weights_source.stat().st_size > 0

    real_input = _canonical_input(example.detach().numpy())
    input_code = bakenn.quantize_input(compiled.plan, real_input)
    output_code = bakenn.run_reference(compiled.plan, input_code)
    dequantized = bakenn.dequantize_output(compiled.plan, output_code)
    expected = model(example).detach().numpy()
    np.testing.assert_allclose(dequantized, expected, rtol=0.0, atol=case.fp_tolerance)

    compilers = [path for name in ("gcc", "clang") if (path := shutil.which(name))]
    if not compilers:
        pytest.skip("neither GCC nor Clang is available")
    input_type = compiled.graph.values[compiled.graph.inputs[0]]
    rng = np.random.default_rng(seed)
    random_inputs = rng.integers(
        -128,
        128,
        size=(16, *input_type.shape[1:]),
        dtype=np.int16,
    ).astype(np.int8)
    expected_codes = np.concatenate(
        [
            bakenn.run_reference(compiled.plan, sample.reshape(input_type.shape)).reshape(-1)
            for sample in random_inputs
        ]
    )
    for compiler in compilers:
        executable = _compile_runner(compiled, compiled.artifacts.output_dir, compiler)
        process = subprocess.run(
            executable,
            input=random_inputs.tobytes(),
            capture_output=True,
            check=True,
        )
        np.testing.assert_array_equal(
            np.frombuffer(process.stdout, dtype=np.int8),
            expected_codes,
        )
