from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

import bakenn


torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
import torch.nn.functional as functional  # noqa: E402


class _TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.max_pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(2, 3, 3, padding=1)
        self.avg_pool = nn.AvgPool2d(2, count_include_pad=False)
        self.classifier = nn.Linear(12, 4)
        with torch.no_grad():
            self.bn1.weight.copy_(torch.tensor([0.75, -0.5]))
            self.bn1.bias.copy_(torch.tensor([0.1, -0.2]))

    def forward(self, value):  # type: ignore[no-untyped-def]
        value = functional.relu(self.bn1(self.conv1(value)))
        value = self.max_pool(value)
        value = functional.relu(self.conv2(value))
        value = self.avg_pool(value)
        value = self.classifier(torch.flatten(value, 1))
        return torch.softmax(value, dim=-1)


class _ResidualDSCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(2, 2, 1)
        self.depthwise = nn.Conv2d(2, 2, 3, padding=1, groups=2)
        self.project = nn.Conv2d(2, 2, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2, 3)

    def forward(self, value):  # type: ignore[no-untyped-def]
        residual = self.stem(value)
        value = self.project(self.depthwise(residual))
        value += residual
        value = functional.relu(value)
        value = self.global_pool(value)
        return self.classifier(torch.flatten(value, 1))


class _MobileBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.expand = nn.Conv2d(2, 4, 1)
        self.depthwise = nn.Conv2d(4, 4, 3, padding=1, groups=4)
        self.project = nn.Conv2d(4, 2, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2, 3)

    def forward(self, value):  # type: ignore[no-untyped-def]
        residual = value
        value = functional.relu6(self.expand(value))
        value = functional.relu6(self.depthwise(value))
        value = self.project(value)
        value += residual
        value = self.global_pool(value)
        return self.classifier(torch.flatten(value, 1))


class _MobileNetV3Mini(nn.Module):
    """Small complete classifier exercising the MobileNetV3-specific surface."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(4)
        self.depthwise = nn.Conv2d(4, 4, 3, padding=1, groups=4, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(4)
        self.se_reduce = nn.Conv2d(4, 2, 1)
        self.se_expand = nn.Conv2d(2, 4, 1)
        self.project = nn.Conv2d(4, 4, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(4, 3)

    def forward(self, value):  # type: ignore[no-untyped-def]
        value = functional.hardswish(self.stem_bn(self.stem(value)))
        value = functional.hardswish(self.depthwise_bn(self.depthwise(value)))
        scale = self.pool(value)
        scale = functional.relu(self.se_reduce(scale))
        scale = functional.hardsigmoid(self.se_expand(scale))
        value = self.project(value * scale)
        return self.classifier(torch.flatten(self.pool(value), 1))


class _EfficientNetLiteMini(nn.Module):
    """EfficientNet-Lite style inverted bottleneck with no SE or SiLU."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 3, padding=1)
        self.expand = nn.Conv2d(4, 8, 1)
        self.depthwise = nn.Conv2d(8, 8, 3, padding=1, groups=8)
        self.project = nn.Conv2d(8, 4, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(4, 3)

    def forward(self, value):  # type: ignore[no-untyped-def]
        value = functional.relu6(self.stem(value))
        residual = value
        value = functional.relu6(self.expand(value))
        value = functional.relu6(self.depthwise(value))
        value = self.project(value) + residual
        return self.classifier(torch.flatten(self.pool(value), 1))


class _UNetTransposeMini(nn.Module):
    """Compact U-Net with transpose-convolution upsampling and a skip concat."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, 3, padding=1)
        self.bottleneck = nn.Conv2d(4, 8, 3, padding=1)
        self.up = nn.ConvTranspose2d(8, 4, 2, stride=2)
        self.decoder = nn.Conv2d(8, 4, 3, padding=1)
        self.output = nn.Conv2d(4, 2, 1)

    def forward(self, value):  # type: ignore[no-untyped-def]
        skip = functional.relu(self.encoder(value))
        value = functional.max_pool2d(skip, 2)
        value = functional.relu(self.bottleneck(value))
        value = self.up(value)
        value = torch.cat((value, skip), dim=1)
        return self.output(functional.relu(self.decoder(value)))


class _UNetResizeMini(nn.Module):
    """Compact U-Net using static bilinear resize instead of ConvTranspose2d."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, 3, padding=1)
        self.bottleneck = nn.Conv2d(4, 8, 3, padding=1)
        self.reduce = nn.Conv2d(8, 4, 1)
        self.decoder = nn.Conv2d(8, 2, 3, padding=1)

    def forward(self, value):  # type: ignore[no-untyped-def]
        skip = functional.relu(self.encoder(value))
        value = functional.max_pool2d(skip, 2)
        value = functional.relu(self.bottleneck(value))
        value = self.reduce(value)
        value = functional.interpolate(
            value,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.decoder(torch.cat((value, skip), dim=1))


class _UNetNearestMini(_UNetResizeMini):
    """Nearest-neighbor variant sharing the same static U-Net decoder."""

    def forward(self, value):  # type: ignore[no-untyped-def]
        skip = functional.relu(self.encoder(value))
        value = functional.max_pool2d(skip, 2)
        value = functional.relu(self.bottleneck(value))
        value = self.reduce(value)
        value = functional.interpolate(value, size=skip.shape[-2:], mode="nearest")
        return self.decoder(torch.cat((value, skip), dim=1))


class _UNetGroupedCropMini(nn.Module):
    """Odd-resolution U-Net exercising grouped deconvolution and static crop."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, 3, padding=1)
        self.bottleneck = nn.Conv2d(4, 4, 3, padding=1)
        self.up = nn.ConvTranspose2d(4, 4, 2, stride=2, groups=2)
        self.decoder = nn.Conv2d(8, 2, 3, padding=1)

    def forward(self, value):  # type: ignore[no-untyped-def]
        skip = functional.relu(self.encoder(value))
        value = functional.max_pool2d(skip, 2)
        value = functional.relu(self.bottleneck(value))
        value = self.up(value)
        skip = skip[:, :, 1:9, 1:9]
        return self.decoder(torch.cat((value, skip), dim=1))


@dataclass(frozen=True)
class _Case:
    name: str
    model_type: type[nn.Module]
    input_shape: tuple[int, ...]
    seed: int
    fp_tolerance: float
    calibration_samples: int = 24
    required_float_ops: tuple[str, ...] = ()
    required_plan_operations: tuple[str, ...] = ()


_CASES = (
    _Case("torch_tiny_cnn", _TinyCNN, (1, 1, 8, 8), 4101, 0.04),
    _Case("torch_residual_ds_cnn", _ResidualDSCNN, (1, 2, 4, 4), 4102, 0.08),
    _Case("torch_mobile_block", _MobileBlock, (1, 2, 4, 4), 4103, 0.08),
    _Case(
        "torch_mobilenet_v3_mini",
        _MobileNetV3Mini,
        (1, 3, 8, 8),
        4104,
        0.16,
        8,
        ("FloatHardSwishOp", "FloatHardSigmoidOp", "FloatMulOp"),
        ("hardswish", "hardsigmoid"),
    ),
    _Case(
        "torch_efficientnet_lite_mini",
        _EfficientNetLiteMini,
        (1, 3, 8, 8),
        4105,
        0.16,
        8,
        ("FloatDepthwiseConv2DOp", "FloatAddOp", "FloatReLU6Op"),
    ),
    _Case(
        "torch_unet_transpose_mini",
        _UNetTransposeMini,
        (1, 3, 8, 8),
        4106,
        0.20,
        8,
        ("FloatConvTranspose2DOp", "FloatConcatOp"),
    ),
    _Case(
        "torch_unet_resize_mini",
        _UNetResizeMini,
        (1, 3, 8, 8),
        4107,
        0.20,
        8,
        ("FloatResizeBilinear2DOp", "FloatConcatOp"),
    ),
    _Case(
        "torch_unet_nearest_mini",
        _UNetNearestMini,
        (1, 3, 8, 8),
        4108,
        0.20,
        8,
        ("FloatResizeNearest2DOp", "FloatConcatOp"),
    ),
    _Case(
        "torch_unet_grouped_crop_mini",
        _UNetGroupedCropMini,
        (1, 3, 9, 9),
        4109,
        0.25,
        8,
        ("FloatConvTranspose2DOp", "FloatSliceOp", "FloatConcatOp"),
    ),
)


def _compile_runner(artifacts, directory: Path, compiler: str) -> Path:  # type: ignore[no-untyped-def]
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    symbol = manifest["model"]
    macro = symbol.upper()
    runner = directory / f"runner_{compiler}.c"
    runner.write_text(
        f'''#include "{artifacts.header.name}"
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#define GUARD_SIZE 32u

static int guard_ok(const uint8_t *value) {{
    for (size_t index = 0; index < GUARD_SIZE; ++index) {{
        if (value[index] != UINT8_C(0xA5)) {{ return 0; }}
    }}
    return 1;
}}

int main(void) {{
    _Alignas({macro}_ARENA_ALIGNMENT)
        uint8_t arena_storage[GUARD_SIZE + {macro}_ARENA_SIZE + GUARD_SIZE];
    struct {{
        uint8_t before[GUARD_SIZE];
        int8_t data[{macro}_INPUT_SIZE];
        uint8_t after[GUARD_SIZE];
    }} input;
    struct {{
        uint8_t before[GUARD_SIZE];
        int8_t data[{macro}_OUTPUT_SIZE];
        uint8_t after[GUARD_SIZE];
    }} output;
    memset(arena_storage, 0xA5, sizeof(arena_storage));
    memset(&input, 0xA5, sizeof(input));
    memset(&output, 0xA5, sizeof(output));
    uint8_t *arena = {macro}_ARENA_SIZE == 0u ? NULL : arena_storage + GUARD_SIZE;
    while (fread(input.data, 1u, {macro}_INPUT_BYTES, stdin) == {macro}_INPUT_BYTES) {{
        {symbol}_infer(arena, input.data, output.data);
        if (!guard_ok(arena_storage)
            || !guard_ok(arena_storage + GUARD_SIZE + {macro}_ARENA_SIZE)
            || !guard_ok(input.before) || !guard_ok(input.after)
            || !guard_ok(output.before) || !guard_ok(output.after)) {{
            return 4;
        }}
        if (fwrite(output.data, 1u, {macro}_OUTPUT_BYTES, stdout) != {macro}_OUTPUT_BYTES) {{
            return 2;
        }}
    }}
    return ferror(stdin) ? 3 : 0;
}}
''',
        encoding="utf-8",
    )
    executable = directory / f"runner_{compiler}"
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
            str(artifacts.model_source),
            str(artifacts.weights_source),
            str(artifacts.kernels_source),
            str(runner),
            "-I",
            str(directory),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
    )
    return executable


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_fp32_torch_ptq_to_generated_c_is_bit_exact_and_accurate(
    tmp_path: Path,
    case: _Case,
) -> None:
    torch.manual_seed(case.seed)
    model = case.model_type().eval()
    calibration = torch.randn(
        case.calibration_samples,
        *case.input_shape[1:],
        dtype=torch.float32,
    )
    example = calibration[:1]
    output_dir = tmp_path / case.name
    compiled = bakenn.compile_torch_ptq(
        model,
        example,
        calibration,
        output_dir,
        name=case.name,
    )

    assert compiled.float_graph.name == case.name
    assert compiled.graph.name == case.name
    assert compiled.plan.name == case.name
    assert compiled.artifacts.header.exists()
    assert compiled.artifacts.manifest.exists()
    float_op_names = {type(op).__name__ for op in compiled.float_graph.ops}
    assert set(case.required_float_ops) <= float_op_names
    plan_operations = {
        operation
        for step in compiled.plan.steps
        if (operation := getattr(step, "operation", None)) is not None
    }
    assert set(case.required_plan_operations) <= plan_operations

    # Firmware input is the declared canonical NHWC layout.  Quantize a real
    # calibration sample through the public ABI contract and compare its
    # dequantized integer result with the original FP32 model.
    sample_nhwc = np.ascontiguousarray(example.detach().numpy().transpose(0, 2, 3, 1))
    sample_code = bakenn.quantize_input(compiled.plan, sample_nhwc)
    reference_code = bakenn.run_reference(compiled.plan, sample_code)
    dequantized = bakenn.dequantize_output(compiled.plan, reference_code)
    expected_float = model(example).detach().numpy()
    if expected_float.ndim == 4:
        expected_float = np.ascontiguousarray(expected_float.transpose(0, 2, 3, 1))
    np.testing.assert_allclose(
        dequantized,
        expected_float,
        rtol=0.0,
        atol=case.fp_tolerance,
    )

    rng = np.random.default_rng(case.seed)
    input_shape = compiled.plan.tensors[compiled.plan.inputs[0]].tensor_type.shape
    input_codes = rng.integers(
        -128,
        128,
        size=(20, *input_shape[1:]),
        dtype=np.int16,
    ).astype(np.int8)
    expected_codes = np.concatenate(
        [
            bakenn.run_reference(compiled.plan, value.reshape(input_shape))
            for value in input_codes
        ],
        axis=0,
    )

    available_compilers = [name for name in ("gcc", "clang") if shutil.which(name)]
    if not available_compilers:
        pytest.skip("neither GCC nor Clang is installed")
    for compiler in available_compilers:
        executable = _compile_runner(compiled.artifacts, output_dir, compiler)
        completed = subprocess.run(
            executable,
            input=input_codes.tobytes(),
            capture_output=True,
            check=True,
        )
        actual = np.frombuffer(completed.stdout, dtype=np.int8).reshape(expected_codes.shape)
        np.testing.assert_array_equal(actual, expected_codes)


@pytest.mark.parametrize(
    ("factory_name", "required_float_ops", "minimum_plan_steps"),
    (
        (
            "mobilenet_v3_small",
            {"FloatHardSwishOp", "FloatHardSigmoidOp", "FloatMulOp"},
            100,
        ),
        (
            "mobilenet_v3_large",
            {"FloatHardSwishOp", "FloatHardSigmoidOp", "FloatMulOp"},
            100,
        ),
        (
            "mobilenet_v2",
            {"FloatDepthwiseConv2DOp", "FloatReLU6Op", "FloatAddOp"},
            60,
        ),
        (
            # torchvision does not ship Google's Lite checkpoint.  B0 is a
            # semantic superset here because it retains SiLU and SE broadcast.
            "efficientnet_b0",
            {"FloatSiLUOp", "FloatSigmoidOp", "FloatMulOp"},
            100,
        ),
        (
            "mnasnet0_5",
            {"FloatDepthwiseConv2DOp", "FloatAddOp", "FloatReduceMeanOp"},
            60,
        ),
    ),
)
def test_unmodified_torchvision_backbone_reaches_generated_c(
    tmp_path: Path,
    factory_name: str,
    required_float_ops: set[str],
    minimum_plan_steps: int,
) -> None:
    try:
        import torchvision.models as models
    except (ImportError, RuntimeError) as error:
        pytest.skip(f"torchvision is unavailable: {error}")

    torch.manual_seed(20260814)
    model = getattr(models, factory_name)(weights=None).eval()
    sample = torch.randn(1, 3, 32, 32)
    compiled = bakenn.compile_torch_ptq(
        model,
        sample,
        [sample],
        tmp_path / factory_name,
        name=factory_name,
    )
    captured = {type(op).__name__ for op in compiled.float_graph.ops}
    assert required_float_ops <= captured
    assert len(compiled.plan.steps) >= minimum_plan_steps
    assert compiled.artifacts.model_source.stat().st_size > 0
    assert compiled.artifacts.weights_source.stat().st_size > 0


def test_unmodified_torchvision_mobilenet_v2_generated_c_is_byte_exact(
    tmp_path: Path,
) -> None:
    """Run a small raw-int8 corpus through the complete V2 generated C ABI."""

    try:
        import torchvision.models as models
    except (ImportError, RuntimeError) as error:
        pytest.skip(f"torchvision is unavailable: {error}")
    if shutil.which("cc") is None:
        pytest.skip("host C compiler is unavailable")
    compiler = "cc"

    torch.manual_seed(20260815)
    model = models.mobilenet_v2(weights=None).eval()
    sample = torch.randn(1, 3, 32, 32)
    compiled = bakenn.compile_torch_ptq(
        model,
        sample,
        [sample],
        tmp_path / "mobilenet_v2_c",
        name="mobilenet_v2_c",
    )
    executable = _compile_runner(compiled.artifacts, tmp_path / "mobilenet_v2_c", compiler)
    input_type = compiled.graph.values[compiled.graph.inputs[0]]
    rng = np.random.default_rng(20260815)
    inputs = rng.integers(
        -128,
        128,
        size=(2, *input_type.shape[1:]),
        dtype=np.int16,
    ).astype(np.int8)
    expected = np.concatenate(
        [
            bakenn.run_reference(compiled.plan, item.reshape(input_type.shape))
            for item in inputs
        ],
        axis=0,
    )
    process = subprocess.run(
        executable,
        input=inputs.tobytes(),
        capture_output=True,
        check=True,
    )
    np.testing.assert_array_equal(
        np.frombuffer(process.stdout, dtype=np.int8).reshape(expected.shape),
        expected,
    )
