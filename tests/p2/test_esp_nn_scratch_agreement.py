"""The host scratch mirror must equal the vendored ESP-NN S3 getters.

BakeNN sizes the shared ESP-NN scratch region at compile time from a Python
mirror of the pinned getter arithmetic.  If the mirror under-reports, a kernel
writes past its planned region on the target.  These tests compare the mirror
with the real vendored getters, compiled for the host, over a sweep that
reaches every getter branch and over every Conv2D/DepthwiseConv2D layer of the
representative MobileNetV2-0.25 model.
"""

from __future__ import annotations

from collections import Counter
import importlib.util
import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest

import bakenn
from bakenn.backend.esp_nn.integration import (
    _esp32s3_conv_scratch,
    _esp32s3_depthwise_scratch,
)
from bakenn.plan.steps import Conv2DStep, DepthwiseConv2DStep
from bakenn.targets import ESP32_S3

from .esp_nn_scratch_probe import (
    ScratchCase,
    build_s3_scratch_probe,
    case_for_step,
    mismatches,
    query_s3_scratch,
)


@pytest.fixture(scope="module")
def probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_s3_scratch_probe(tmp_path_factory.mktemp("esp_nn_s3_scratch"))


def _extent(size: int, kernel: int, stride: int, before: int, after: int) -> int | None:
    padded = size + before + after
    if padded < kernel:
        return None
    return (padded - kernel) // stride + 1


def _mirror(case: ScratchCase) -> int:
    step = SimpleNamespace(
        padding=case.padding,
        stride=case.stride,
        depth_multiplier=case.count,
    )
    input_shape = (1, case.input_height, case.input_width, case.input_channels)
    if case.kind == "conv":
        return _esp32s3_conv_scratch(
            step,  # type: ignore[arg-type]
            input_shape,
            (1, case.output_height, case.output_width, case.count),
            (case.count, case.kernel_height, case.kernel_width, case.input_channels),
        )
    channels = case.input_channels * case.count
    return _esp32s3_depthwise_scratch(
        step,  # type: ignore[arg-type]
        input_shape,
        (1, case.output_height, case.output_width, channels),
        (case.kernel_height, case.kernel_width, channels),
    )


def _cases(
    kind: str,
    *,
    sizes: tuple[tuple[int, int], ...],
    kernels: tuple[tuple[int, int], ...],
    channels: tuple[int, ...],
    counts: tuple[int, ...],
    strides: tuple[tuple[int, int], ...],
    paddings: tuple[tuple[int, int, int, int], ...],
) -> list[ScratchCase]:
    cases = []
    for (height, width), (kernel_height, kernel_width), input_channels, count, stride, padding in (
        itertools.product(sizes, kernels, channels, counts, strides, paddings)
    ):
        output_height = _extent(height, kernel_height, stride[0], padding[0], padding[1])
        output_width = _extent(width, kernel_width, stride[1], padding[2], padding[3])
        if output_height is None or output_width is None:
            continue
        cases.append(
            ScratchCase(
                kind,
                height,
                width,
                input_channels,
                kernel_height,
                kernel_width,
                count,
                output_height,
                output_width,
                stride,
                padding,
            )
        )
    return cases


def _conv_branch(case: ScratchCase) -> str:
    """Name the esp_nn_get_conv_scratch_size_esp32s3 branch a case reaches."""

    pad_top, _, pad_left, _ = case.padding
    if (
        (case.kernel_height, case.kernel_width) == (1, 1)
        and pad_top == pad_left == 0
        and case.stride == (1, 1)
    ):
        layout = "channel_padded" if case.input_channels % 8 else "aligned"
        area = "small" if case.input_width * case.input_height < 8 else "transposed"
        return f"pointwise_{layout}_{area}"
    filter_row = case.kernel_width * case.input_channels
    window = filter_row * case.kernel_height
    if filter_row < 16 and window >= 16:
        return "im2col"
    stride_height, stride_width = case.stride
    pad_right = max(
        0,
        (case.output_width - 1) * stride_width + case.kernel_width - pad_left
        - case.input_width,
    )
    pad_bottom = max(
        0,
        (case.output_height - 1) * stride_height + case.kernel_height - pad_top
        - case.input_height,
    )
    padded = any((pad_top, pad_left, pad_right, pad_bottom))
    return "general_padded" if padded else "general_unpadded"


def _depthwise_branch(case: ScratchCase) -> str:
    """Name the esp_nn_get_depthwise_conv_scratch_size_esp32s3 branch."""

    channels, multiplier = case.input_channels, case.count
    pad_top, _, pad_left, _ = case.padding
    stride_height, stride_width = case.stride
    kernel = (case.kernel_height, case.kernel_width)
    if multiplier == 1 and channels % 8 == 0:
        if kernel == (3, 3):
            if channels % 16 == 0:
                if pad_left or pad_top:
                    pad_width, pad_height = 2 * pad_left, 2 * pad_top
                else:
                    pad_width = case.output_width * stride_width + 2 - case.input_width
                    pad_height = case.output_height * stride_height + 2 - case.input_height
                if not (pad_width or pad_height):
                    return "3x3_c16_unpadded"
                full = (
                    (case.input_width + pad_width)
                    * (case.input_height + pad_height)
                    * channels
                )
                return "3x3_c16_padded_full" if full <= 40 * 1024 else "3x3_c16_padded_tiled"
            return "3x3_c8_widened" if channels >= 12 else "3x3_c8"
        total = 2 * (
            case.kernel_height * case.kernel_width * channels
            + case.input_width * case.input_height * channels
        )
        return "generic_c8_full" if total <= 48 * 1024 else "generic_c8_tiled"
    if multiplier == 1 and channels > 3:
        return "multiplier1_channel_padded"
    if multiplier % 4 == 0:
        return "multiplier4"
    return "fallback"


def test_s3_conv_scratch_mirror_matches_vendored_getter_on_every_branch(
    probe: Path,
) -> None:
    cases = _cases(
        "conv",
        sizes=((1, 1), (2, 3), (4, 4), (5, 7), (8, 8), (16, 16), (17, 9)),
        kernels=((1, 1), (3, 3), (5, 5), (1, 3), (3, 1), (2, 2)),
        channels=(1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 24, 32),
        counts=(1, 3, 8, 16),
        strides=((1, 1), (2, 2), (1, 2)),
        paddings=((0, 0, 0, 0), (1, 1, 1, 1), (0, 1, 0, 1), (2, 2, 2, 2)),
    )
    branches = Counter(_conv_branch(case) for case in cases)
    assert set(branches) == {
        "pointwise_aligned_small",
        "pointwise_aligned_transposed",
        "pointwise_channel_padded_small",
        "pointwise_channel_padded_transposed",
        "im2col",
        "general_padded",
        "general_unpadded",
    }, branches

    official = query_s3_scratch(probe, cases)

    assert mismatches(cases, [_mirror(case) for case in cases], official) == []


def test_s3_depthwise_scratch_mirror_matches_vendored_getter_on_every_branch(
    probe: Path,
) -> None:
    cases = _cases(
        "depthwise",
        # 48x48 and 49x49 straddle the 40 KiB padded-input threshold at 16 channels.
        sizes=(
            (1, 1), (3, 3), (4, 4), (7, 7), (8, 8), (16, 16), (32, 32),
            (48, 48), (49, 49), (52, 52),
        ),
        kernels=((3, 3), (5, 5), (1, 1), (3, 1), (2, 2)),
        channels=(1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 40, 48),
        counts=(1, 2, 3, 4, 8),
        strides=((1, 1), (2, 2)),
        # The S3 capability only accepts symmetric depthwise padding.
        paddings=((0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2)),
    )
    branches = Counter(_depthwise_branch(case) for case in cases)
    assert set(branches) == {
        "3x3_c16_unpadded",
        "3x3_c16_padded_full",
        "3x3_c16_padded_tiled",
        "3x3_c8_widened",
        "3x3_c8",
        "generic_c8_full",
        "generic_c8_tiled",
        "multiplier1_channel_padded",
        "multiplier4",
        "fallback",
    }, branches

    official = query_s3_scratch(probe, cases)

    assert mismatches(cases, [_mirror(case) for case in cases], official) == []


def _mobilenet_v2_example():  # type: ignore[no-untyped-def]
    root = Path(__file__).resolve().parents[2]
    source = root / "examples/mobilenet_v2_cifar10/run.py"
    spec = importlib.util.spec_from_file_location("bakenn_mobilenet_scratch_example", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_s3_scratch_for_every_mobilenet_v2_layer_matches_vendored_getter(
    probe: Path,
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    model = _mobilenet_v2_example().mobilenet_v2_quarter().eval()
    generator = torch.Generator().manual_seed(38)
    calibration = [torch.randn(1, 3, 32, 32, generator=generator) for _ in range(4)]
    compiled = bakenn.compile_torch_ptq(
        model,
        calibration[0],
        calibration,
        tmp_path / "mobilenet_v2_s3",
        name="mobilenet_v2_s3",
        backend_options=bakenn.CBackendOptions(
            kernel_policy=bakenn.KernelPolicy.STATIC_PRIORITY,
            enable_esp_nn=True,
            target=ESP32_S3,
        ),
        target=ESP32_S3,
    )
    plan = compiled.plan
    backend_plan = compiled.artifacts.backend_plan
    layers = [
        (step, selection)
        for step, selection in zip(plan.steps, backend_plan.selections)
        if isinstance(step, (Conv2DStep, DepthwiseConv2DStep))
    ]
    cases = [case_for_step(plan, step) for step, _ in layers]
    layer_mix = Counter(
        "depthwise"
        if case.kind == "depthwise"
        else f"conv{case.kernel_height}x{case.kernel_width}"
        for case in cases
    )
    # torchvision MobileNetV2: stem 3x3, 34 pointwise, 17 depthwise layers.
    assert layer_mix == {"conv3x3": 1, "conv1x1": 34, "depthwise": 17}
    not_esp_nn = {
        selection.step_name: selection.kernel_id
        for _, selection in layers
        if not selection.kernel_id.startswith("esp_nn.esp32s3.")
    }
    assert not_esp_nn == {}

    official = query_s3_scratch(probe, cases)
    planned = [selection.scratch_size for _, selection in layers]

    assert mismatches(cases, planned, official) == []
    assert all(selection.scratch_alignment == 16 for _, selection in layers)
    assert backend_plan.scratch_size >= max(official)
