#!/usr/bin/env python3
"""Run BakeNN's numbered functional tests offline; print one result line per test.

Every test checks one thing. The tests use only files in this repository (the
frozen MNIST checkpoint, calibration images and test images under
``examples/mnist/evidence``) and small models built from fixed seeds. Nothing
is downloaded and nothing is trained.

    python scripts/functional_checks.py                # all tests
    python scripts/functional_checks.py T007 T012      # selected tests
    python scripts/functional_checks.py T016-T054      # a range of tests
    python scripts/functional_checks.py --list         # numbers, groups and titles
    python scripts/functional_checks.py --strict       # a missing tool is a failure

A test passes only when its stated condition holds; ``SKIP`` means an optional
host tool (a cross compiler, a C++ compiler, a sanitizer runtime) is not
installed. The same tests run under pytest through
``tests/test_functional_checks.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Callable

REPOSITORY = Path(__file__).resolve().parents[1]
for entry in ("src", "examples/mnist", "examples/targets", "scripts"):
    sys.path.insert(0, str(REPOSITORY / entry))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional  # noqa: E402

import bakenn  # noqa: E402
from bakenn.errors import CompileError  # noqa: E402
from run_mnist import MNISTNet, quantize_mnist_corpus  # noqa: E402

EVIDENCE = REPOSITORY / "examples/mnist/evidence"
WARNINGS = ("-Wall", "-Wextra", "-Werror", "-pedantic")
ENTRY_POINTS = (
    "compile_torch_ptq", "compile", "quantize_input", "dequantize_output", "run_reference",
    "run_reference_trace", "load_manifest", "build_freestanding_elf", "export_esp_idf_project",
    "export_esp_idf_component", "export_zephyr_project", "export_arduino_library",
)
PROFILES = ("portable32", "cortex-m0plus", "cortex-m4", "rv32imc", "esp32", "esp32s3", "esp32c3")
# Unrelated Arm cores, from ARMv6-M without hardware divide to ARMv8.1-M.
ARM_CORES = (
    "cortex-m0plus", "cortex-m3", "cortex-m4", "cortex-m7", "cortex-m33", "cortex-m55", "cortex-m85",
)
RISCV_COMPILERS = (
    "riscv-none-elf-gcc", "riscv64-elf-gcc", "riscv64-unknown-elf-gcc", "riscv32-unknown-elf-gcc",
)
GENERATED_FILES = (
    ("header", "public header"),
    ("model_source", "inference source"),
    ("weights_header", "weights header"),
    ("weights_source", "weights source"),
    ("kernels_header", "kernels header"),
    ("kernels_source", "kernels source"),
    ("manifest", "manifest"),
    ("memory_report_json", "JSON memory report"),
    ("memory_report_text", "text memory report"),
    ("build_fragment", "CMake source list"),
)


class Skip(Exception):
    """An optional host tool needed by this test is not installed."""


@dataclass(frozen=True)
class Check:
    identifier: str
    group: str
    title: str
    run: Callable[["Session"], str]


class Session:
    """Shared inputs and intermediate results, each built once per run."""

    def __init__(self, output: Path, compiler: str) -> None:
        self.output = output
        self.compiler = compiler
        self._cache: dict[str, object] = {}
        # Torch 2.9 logs a recompile warning once several freed modules have
        # been exported. Keeping every module alive keeps the output quiet.
        self._modules: list[nn.Module] = []
        self.evidence = json.loads((EVIDENCE / "mnist_evidence.json").read_text())
        calibration_shape = tuple(self.evidence["calibration"]["shape_nhw"])
        raw = np.fromfile(EVIDENCE / "calibration_images_u8.bin", dtype=np.uint8)
        self.calibration = (
            torch.from_numpy(raw.reshape(calibration_shape).copy()).unsqueeze(1).float() / 255.0
        )

    def once(self, key: str, build: Callable[[], object]):  # type: ignore[no-untyped-def]
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    def keep(self, module: nn.Module) -> nn.Module:
        self._modules.append(module)
        return module

    def mnist(self) -> nn.Module:
        model = MNISTNet().eval()
        model.load_state_dict(
            torch.load(EVIDENCE / "mnist_fp32.pt", map_location="cpu", weights_only=True)
        )
        return self.keep(model)

    def compile(self, name: str, **options: object):  # type: ignore[no-untyped-def]
        """Quantize and compile the frozen MNIST model."""

        return bakenn.compile_torch_ptq(
            self.mnist(),
            self.calibration[:1],
            self.calibration,
            self.output / name,
            name="mnist",
            **options,
        )

    @property
    def compiled(self):  # type: ignore[no-untyped-def]
        return self.once("portable", lambda: self.compile("portable"))

    def backend(self, name: str, **options: object):  # type: ignore[no-untyped-def]
        """Generate C again from the quantized MNIST graph; only the backend runs."""

        return bakenn.compile(self.compiled.graph, self.output / name, **options)


def run_check(check: Check, session: Session) -> str:
    """Run one test and return its evidence line."""

    # Finder drops .DS_Store into a folder that is open on screen, and the
    # manifest check rejects any file it did not hash.
    for stray in session.output.rglob(".DS_Store"):
        stray.unlink(missing_ok=True)
    return check.run(session)


def _require(*tools: str) -> str:
    for tool in tools:
        resolved = shutil.which(tool)
        if resolved is not None:
            return resolved
    raise Skip(f"{tools[0]} is not installed")


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _rejection(action: Callable[[], object], expected: str, kind: type = CompileError) -> Exception:
    """Return the error of an action that must be refused with a known message."""

    try:
        action()
    except kind as error:
        _expect(expected in str(error), f"unexpected rejection: {error}")
        return error
    raise AssertionError("an input that must be rejected was accepted")


def _refused(action: Callable[[], object], expected: str, kind: type = CompileError) -> str:
    _rejection(action, expected, kind)
    return f'rejected=1 message_contains="{expected}"'


def _digest(*paths: Path) -> str:
    return hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()


def _sources(artifacts: object) -> tuple[Path, Path, Path]:
    return (artifacts.model_source, artifacts.weights_source, artifacts.kernels_source)  # type: ignore[attr-defined]


def _runner_source(artifacts: object) -> str:
    """A host program that maps INT8 inputs on stdin to INT8 outputs on stdout."""

    symbol = artifacts.model_source.stem  # type: ignore[attr-defined]
    macro = symbol.upper()
    return f"""#include "{artifacts.header.name}"
#include <stddef.h>
#include <stdio.h>

int main(void) {{
    _Alignas({macro}_ARENA_ALIGNMENT) static uint8_t arena[{macro}_ARENA_SIZE > 0u ? {macro}_ARENA_SIZE : 1u];
    static int8_t input[{macro}_INPUT_SIZE];
    static int8_t output[{macro}_OUTPUT_SIZE];
    while (fread(input, 1u, {macro}_INPUT_BYTES, stdin) == {macro}_INPUT_BYTES) {{
        {symbol}_infer({macro}_ARENA_SIZE > 0u ? arena : NULL, input, output);
        if (fwrite(output, 1u, {macro}_OUTPUT_BYTES, stdout) != {macro}_OUTPUT_BYTES) {{
            return 1;
        }}
    }}
    return 0;
}}
"""


def _guard_source(artifacts: object) -> str:
    """A host program that counts writes outside the buffers the model may write."""

    symbol = artifacts.model_source.stem  # type: ignore[attr-defined]
    macro = symbol.upper()
    return f"""#include "{artifacts.header.name}"
#include <stdio.h>
#include <string.h>

#define GUARD 64u

static unsigned damaged(const uint8_t *guard) {{
    unsigned count = 0u;
    for (unsigned index = 0u; index < GUARD; ++index) {{
        count += guard[index] != 0xA5u;
    }}
    return count;
}}

int main(void) {{
    _Alignas(64) static uint8_t arena[GUARD + {macro}_ARENA_SIZE + GUARD];
    static uint8_t input[GUARD + {macro}_INPUT_BYTES + GUARD];
    static uint8_t output[GUARD + {macro}_OUTPUT_BYTES + GUARD];
    static int8_t pristine[{macro}_INPUT_BYTES];
    static int8_t first[{macro}_OUTPUT_BYTES];
    unsigned images = 0u, guard_bytes = 0u, inputs = 0u, repeats = 0u;
    while (fread(pristine, 1u, {macro}_INPUT_BYTES, stdin) == {macro}_INPUT_BYTES) {{
        memset(arena, 0xA5, sizeof arena);
        memset(input, 0xA5, sizeof input);
        memset(output, 0xA5, sizeof output);
        memcpy(input + GUARD, pristine, {macro}_INPUT_BYTES);
        {symbol}_infer(arena + GUARD, (const int8_t *)(input + GUARD), (int8_t *)(output + GUARD));
        memcpy(first, output + GUARD, {macro}_OUTPUT_BYTES);
        /* The second run starts from a different arena and must not notice. */
        memset(arena + GUARD, 0x5A, {macro}_ARENA_SIZE);
        {symbol}_infer(arena + GUARD, (const int8_t *)(input + GUARD), (int8_t *)(output + GUARD));
        guard_bytes += damaged(arena) + damaged(arena + GUARD + {macro}_ARENA_SIZE);
        guard_bytes += damaged(input) + damaged(input + GUARD + {macro}_INPUT_BYTES);
        guard_bytes += damaged(output) + damaged(output + GUARD + {macro}_OUTPUT_BYTES);
        inputs += memcmp(input + GUARD, pristine, {macro}_INPUT_BYTES) != 0;
        repeats += memcmp(first, output + GUARD, {macro}_OUTPUT_BYTES) != 0;
        ++images;
    }}
    printf("%u %u %u %u\\n", images, guard_bytes, inputs, repeats);
    return 0;
}}
"""


def _build(session: Session, artifacts: object, name: str, *flags: str, source: str | None = None) -> Path:
    """Compile the generated C with a host main, outside the hashed artifact tree."""

    directory = session.output / "host" / name
    directory.mkdir(parents=True, exist_ok=True)
    main = directory / "main.c"
    main.write_text(source if source is not None else _runner_source(artifacts))
    executable = directory / "run"
    subprocess.run(
        [
            session.compiler, *flags, *WARNINGS, "-I", str(artifacts.output_dir),  # type: ignore[attr-defined]
            *map(str, _sources(artifacts)), str(main), "-o", str(executable),
        ],
        check=True,
    )
    return executable


def _outputs(executable: Path, inputs: np.ndarray) -> np.ndarray:
    completed = subprocess.run([str(executable)], input=inputs.tobytes(), capture_output=True, check=True)
    return np.frombuffer(completed.stdout, dtype=np.int8)


def _corpus(session: Session) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The 100 frozen test images: pixels, INT8 inputs, expected outputs, labels."""

    def build() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        shape = tuple(session.evidence["physical_test_corpus"]["input_shape_nhwc"])
        pixels = np.fromfile(EVIDENCE / "physical_test_images_u8.bin", dtype=np.uint8)
        pixels = pixels.reshape(shape[:-1])
        inputs = quantize_mnist_corpus(session.compiled.plan, pixels)
        expected = np.fromfile(EVIDENCE / "physical_expected_outputs_int8.bin", dtype=np.int8)
        labels = np.fromfile(EVIDENCE / "physical_test_labels_u8.bin", dtype=np.uint8)
        return pixels, inputs, expected.reshape(shape[0], -1), labels

    return session.once("corpus", build)


def _mnist_outputs(session: Session, name: str, *flags: str) -> np.ndarray:
    """Outputs of the generated MNIST C for the frozen images, built with the given flags."""

    def build() -> np.ndarray:
        _, inputs, expected, _ = _corpus(session)
        executable = _build(session, session.compiled.artifacts, name, "-std=c11", *flags)
        return _outputs(executable, inputs).reshape(expected.shape)

    return session.once(f"host.{name}", build)


def _mismatched(session: Session, name: str, *flags: str) -> str:
    actual = _mnist_outputs(session, name, *flags)
    expected = _corpus(session)[2]
    mismatched = int(np.count_nonzero(actual != expected))
    _expect(mismatched == 0, f"{mismatched} bytes differ from the frozen expected outputs")
    return f"mismatched_bytes={mismatched}/{actual.size}"


def _manifest(artifacts: object) -> dict:
    return json.loads(artifacts.manifest.read_text())  # type: ignore[attr-defined]


def _kernels(artifacts: object) -> list[str]:
    return [item.kernel_id for item in artifacts.backend_plan.selections]  # type: ignore[attr-defined]


def _header_macros(artifacts: object) -> dict[str, str]:
    prefix = artifacts.model_source.stem.upper()  # type: ignore[attr-defined]
    text = artifacts.header.read_text()  # type: ignore[attr-defined]
    return dict(re.findall(rf"^#define {prefix}_(\w+) (\S+)$", text, flags=re.M))


# --- Small seeded models -----------------------------------------------------


class _Lambda(nn.Module):
    """A module whose forward is a function of the input and its layers."""

    def __init__(self, function: Callable[..., object], *layers: nn.Module) -> None:
        super().__init__()
        self.function = function
        self.layers = nn.ModuleList(layers)

    def forward(self, value):  # type: ignore[no-untyped-def]
        return self.function(value, *self.layers)


def _conv(inputs: int = 3, outputs: int = 4, **options: object) -> nn.Conv2d:
    options.setdefault("padding", 1)
    return nn.Conv2d(inputs, outputs, 3, **options)  # type: ignore[arg-type]


def _conv1d() -> nn.Conv1d:
    return nn.Conv1d(4, 6, 3, padding=1)


IMAGE = (1, 3, 8, 8)
# (name in the title, directory and C symbol, model factory, input shape, kernel that must be selected)
OPERATORS = (
    ("Conv2d", "conv2d", lambda: nn.Sequential(_conv()), IMAGE, "conv2d_s8"),
    ("Conv2d with stride 2", "conv2d_stride", lambda: nn.Sequential(_conv(stride=2)), IMAGE, "conv2d_s8"),
    ("Conv2d with dilation 2", "conv2d_dilation", lambda: nn.Sequential(_conv(dilation=2, padding=2)), IMAGE, "conv2d_s8"),
    ("1x1 Conv2d", "conv2d_pointwise", lambda: nn.Sequential(nn.Conv2d(3, 5, 1)), IMAGE, "conv2d_s8"),
    ("depthwise Conv2d", "conv2d_depthwise", lambda: nn.Sequential(_conv(4, 4, groups=4)), (1, 4, 8, 8), "depthwise_conv2d_s8"),
    ("grouped Conv2d", "conv2d_grouped", lambda: nn.Sequential(_conv(4, 6, groups=2)), (1, 4, 8, 8), "conv2d_s8"),
    ("Conv1d", "conv1d", lambda: nn.Sequential(_conv1d()), (1, 4, 16), "conv1d_s8"),
    ("ConvTranspose2d", "conv_transpose2d", lambda: nn.Sequential(nn.ConvTranspose2d(4, 3, 2, stride=2)), (1, 4, 4, 4), "conv_transpose2d_s8"),
    ("Linear", "linear", lambda: nn.Sequential(nn.Linear(16, 5)), (1, 16), "linear_s8"),
    ("ReLU", "relu", lambda: nn.Sequential(_conv(), nn.ReLU()), IMAGE, "conv2d_s8"),
    ("ReLU6", "relu6", lambda: nn.Sequential(_conv(), nn.ReLU6()), IMAGE, "conv2d_s8"),
    ("Hardswish", "hardswish", lambda: nn.Sequential(_conv(), nn.Hardswish()), IMAGE, "activation_lut_s8"),
    ("Hardsigmoid", "hardsigmoid", lambda: nn.Sequential(_conv(), nn.Hardsigmoid()), IMAGE, "activation_lut_s8"),
    ("SiLU", "silu", lambda: nn.Sequential(_conv(), nn.SiLU()), IMAGE, "activation_lut_s8"),
    ("Sigmoid", "sigmoid", lambda: nn.Sequential(_conv(), nn.Sigmoid()), IMAGE, "activation_lut_s8"),
    ("Softmax", "softmax", lambda: nn.Sequential(nn.Linear(16, 5), nn.Softmax(dim=-1)), (1, 16), "softmax_s8_q15"),
    ("MaxPool2d", "max_pool2d", lambda: nn.Sequential(_conv(), nn.MaxPool2d(2)), IMAGE, "max_pool2d_s8"),
    ("MaxPool1d", "max_pool1d", lambda: nn.Sequential(_conv1d(), nn.MaxPool1d(2)), (1, 4, 16), "max_pool1d_s8"),
    ("AvgPool2d", "avg_pool2d", lambda: nn.Sequential(_conv(), nn.AvgPool2d(2)), IMAGE, "average_pool2d_s8"),
    ("AvgPool1d", "avg_pool1d", lambda: nn.Sequential(_conv1d(), nn.AvgPool1d(2)), (1, 4, 16), "average_pool1d_s8"),
    ("global average pooling", "global_avg_pool", lambda: nn.Sequential(_conv(), nn.AdaptiveAvgPool2d(1)), IMAGE, "average_pool2d_s8"),
    ("a spatial mean", "mean", lambda: _Lambda(lambda value, conv: conv(value).mean((2, 3), keepdim=True), _conv()), IMAGE, "reduce_mean_s8"),
    ("a residual Add", "add", lambda: _Lambda(lambda value, conv: conv(value) + value, _conv(4, 4)), (1, 4, 8, 8), "add_s8"),
    ("an element-wise Mul", "mul", lambda: _Lambda(lambda value, left, right: left(value) * right(value), _conv(), _conv()), IMAGE, "mul_s8"),
    ("a channel concatenation", "concat", lambda: _Lambda(lambda value, left, right: torch.cat([left(value), right(value)], dim=1), _conv(), _conv()), IMAGE, "concatenate_s8"),
    ("BatchNorm2d folded into Conv2d", "batch_norm", lambda: nn.Sequential(_conv(bias=False), nn.BatchNorm2d(4), nn.ReLU()), IMAGE, "conv2d_s8"),
    ("zero padding", "pad", lambda: _Lambda(lambda value, conv: functional.pad(conv(value), (1, 2, 1, 0)), _conv()), IMAGE, "pad2d_s8"),
    ("Flatten", "flatten", lambda: nn.Sequential(_conv(), nn.Flatten(), nn.Linear(256, 5)), IMAGE, "flatten_view"),
    ("reshape", "reshape", lambda: _Lambda(lambda value, conv, linear: linear(conv(value).reshape(1, -1)), _conv(), nn.Linear(256, 5)), IMAGE, "reshape_view"),
    ("view", "view", lambda: _Lambda(lambda value, conv, linear: linear(conv(value).view(1, -1)), _conv(), nn.Linear(256, 5)), IMAGE, "reshape_view"),
    ("squeeze and unsqueeze", "squeeze", lambda: _Lambda(lambda value: value.squeeze(2).unsqueeze(2)), (1, 4, 1, 5), "reshape_view"),
    ("a static slice", "slice", lambda: _Lambda(lambda value, conv: conv(value)[:, :, 1:5, 2:6], _conv()), IMAGE, "slice_s8"),
    ("nearest-neighbor upsampling", "upsample_nearest", lambda: _Lambda(lambda value, conv: functional.interpolate(conv(value), scale_factor=2, mode="nearest"), _conv()), IMAGE, "resize_nearest2d_s8"),
    ("bilinear upsampling", "upsample_bilinear", lambda: _Lambda(lambda value, conv: functional.interpolate(conv(value), scale_factor=2, mode="bilinear", align_corners=False), _conv()), IMAGE, "resize_bilinear2d_s8"),
    ("Dropout in eval mode", "dropout", lambda: nn.Sequential(_conv(), nn.Dropout(0.5), nn.ReLU()), IMAGE, "conv2d_s8"),
)


def _residual_cnn() -> nn.Module:
    def forward(value, stem, body, head):  # type: ignore[no-untyped-def]
        value = torch.relu(stem(value))
        value = torch.relu(body(value) + value)
        return head(functional.adaptive_avg_pool2d(value, 1).flatten(1))

    return _Lambda(forward, _conv(3, 8), _conv(8, 8), nn.Linear(8, 4))


def _separable_cnn() -> nn.Module:
    return nn.Sequential(
        _conv(3, 8, stride=2), nn.BatchNorm2d(8), nn.ReLU6(),
        _conv(8, 8, groups=8), nn.BatchNorm2d(8), nn.ReLU6(),
        nn.Conv2d(8, 16, 1), nn.ReLU6(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 4),
    )


def _audio_cnn() -> nn.Module:
    return nn.Sequential(
        nn.Conv1d(4, 8, 3, padding=1), nn.BatchNorm1d(8), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(8, 8, 3, padding=1), nn.ReLU(), nn.AvgPool1d(2),
        nn.Flatten(), nn.Linear(8 * 8, 4),
    )


def _mlp() -> nn.Module:
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 10), nn.Softmax(dim=-1))


NETWORKS = (
    ("a residual CNN", "net_residual", _residual_cnn, (1, 3, 16, 16)),
    ("a depthwise-separable CNN", "net_separable", _separable_cnn, (1, 3, 16, 16)),
    ("a 1-D convolutional network", "net_audio", _audio_cnn, (1, 4, 32)),
    ("a multilayer perceptron with Softmax", "net_mlp", _mlp, (1, 16)),
)


def _seeded(session: Session, factory: Callable[[], nn.Module], seed: int) -> nn.Module:
    """Build a model with seeded weights without touching the caller's random state."""

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = factory().eval()
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    module.running_mean.uniform_(-0.5, 0.5)
                    module.running_var.uniform_(0.5, 1.5)
                    module.weight.uniform_(0.5, 1.5)
                    module.bias.uniform_(-0.5, 0.5)
    return session.keep(model)


def _small_model(session: Session, name: str, factory: Callable[[], nn.Module], shape: tuple[int, ...]):  # type: ignore[no-untyped-def]
    """Compile one small seeded model and compare its C output with the reference."""

    def build():  # type: ignore[no-untyped-def]
        seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        model = _seeded(session, factory, seed)
        calibration = torch.randn(8, *shape[1:], generator=torch.Generator().manual_seed(seed))
        compiled = bakenn.compile_torch_ptq(
            model, calibration[:1], calibration, session.output / "models" / name, name=name
        )
        executable = _build(session, compiled.artifacts, name, "-std=c11", "-O1")
        input_shape = compiled.plan.tensors[compiled.plan.inputs[0]].tensor_type.shape
        samples = np.random.default_rng(seed).integers(
            -128, 128, size=(8, *input_shape[1:]), dtype=np.int16
        ).astype(np.int8)
        expected = np.concatenate(
            [bakenn.run_reference(compiled.plan, sample.reshape(input_shape)).reshape(-1) for sample in samples]
        )
        actual = _outputs(executable, samples)
        _expect(actual.size == expected.size, f"{name}: C wrote {actual.size} bytes, expected {expected.size}")
        _expect(np.unique(expected).size > 1, f"{name}: the reference output is constant")
        return compiled, int(np.count_nonzero(actual != expected)), int(expected.size)

    return session.once(f"model.{name}", build)


def check_operator(name: str, factory: Callable[[], nn.Module], shape: tuple[int, ...], kernel: str, session: Session) -> str:
    compiled, mismatched, total = _small_model(session, name, factory, shape)
    kernels = [item.split(".")[1] for item in _kernels(compiled.artifacts)]
    _expect(kernel in kernels, f"{name}: kernel {kernel} was not selected: {kernels}")
    _expect(mismatched == 0, f"{name}: {mismatched} bytes differ from the Python reference")
    return f"kernels={','.join(kernels)} mismatched_bytes={mismatched}/{total}"


def check_network(name: str, factory: Callable[[], nn.Module], shape: tuple[int, ...], session: Session) -> str:
    compiled, mismatched, total = _small_model(session, name, factory, shape)
    _expect(mismatched == 0, f"{name}: {mismatched} bytes differ from the Python reference")
    return f"steps={len(compiled.plan.steps)} mismatched_bytes={mismatched}/{total}"


def _tiny(session: Session):  # type: ignore[no-untyped-def]
    """A small valid model with its example input and calibration batch."""

    def build():  # type: ignore[no-untyped-def]
        model = _seeded(
            session,
            lambda: nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(64, 10)),
            7,
        )
        generator = torch.Generator().manual_seed(7)
        return model, torch.rand(1, 1, 8, 8, generator=generator), torch.rand(16, 1, 8, 8, generator=generator)

    return session.once("tiny", build)


def _rejected_compile(session: Session, name: str, expected: str, **changes: object) -> str:
    """Compile the small valid model with one argument replaced by an invalid one."""

    model, example, calibration = _tiny(session)
    output = session.output / "rejected" / name
    arguments = {"model": model, "example": example, "calibration": calibration, **changes}
    evidence = _refused(
        lambda: bakenn.compile_torch_ptq(
            arguments["model"], arguments["example"], arguments["calibration"], output
        ),
        expected,
    )
    _expect(not output.exists(), "a rejected compile left artifacts behind")
    return evidence


# --- Installation ------------------------------------------------------------


def check_version(session: Session) -> str:
    del session
    _expect(re.match(r"\d+\.\d+\.\d+", bakenn.__version__) is not None, "version is not MAJOR.MINOR.PATCH")
    return f"bakenn={bakenn.__version__}"


def check_runtime(session: Session) -> str:
    del session
    return f"python={sys.version.split()[0]} torch={torch.__version__.split('+')[0]} numpy={np.__version__}"


def check_entry_points(session: Session) -> str:
    del session
    present = [name for name in ENTRY_POINTS if callable(getattr(bakenn, name, None))]
    _expect(len(present) == len(ENTRY_POINTS), f"missing entry points: {set(ENTRY_POINTS) - set(present)}")
    return f"entry_points={len(present)}/{len(ENTRY_POINTS)}"


def check_contract_versions(session: Session) -> str:
    header = session.compiled.artifacts.header.read_text()
    versions = (
        ("MANIFEST_SCHEMA", bakenn.MANIFEST_SCHEMA_VERSION),
        ("C_ABI", bakenn.GENERATED_C_ABI_VERSION),
        ("ARITHMETIC_PROFILE", bakenn.ARITHMETIC_PROFILE_VERSION),
    )
    for name, value in versions:
        _expect(f"#define BKNN_{name}_VERSION {value}u" in header, f"header lacks {name} version {value}")
    return (
        f"manifest_schema={versions[0][1]} c_abi={versions[1][1]} arithmetic_profile={versions[2][1]} "
        f"header_macros_match={len(versions)}/{len(versions)}"
    )


def check_profiles(session: Session) -> str:
    del session
    present = [name for name in PROFILES if name in bakenn.TARGET_PROFILES]
    _expect(len(present) == len(PROFILES), f"missing target profiles: {set(PROFILES) - set(present)}")
    return f"target_profiles={len(present)}/{len(PROFILES)}"


# --- Compiling the MNIST model -----------------------------------------------


def check_plan(session: Session) -> str:
    steps = len(session.compiled.plan.steps)
    _expect(steps > 0, "the execution plan is empty")
    return f"execution_steps={steps}"


def check_layout(session: Session) -> str:
    plan = session.compiled.plan
    shape = plan.tensors[plan.inputs[0]].tensor_type.shape
    macros = _header_macros(session.compiled.artifacts)
    _expect(tuple(shape) == (1, 28, 28, 1), f"unexpected C input shape {shape}")
    _expect(macros["INPUT_LAYOUT"] == "BKNN_LAYOUT_NHWC", "the header does not declare an NHWC input")
    return "torch_input=1x1x28x28 c_input=1x28x28x1 layout=NHWC"


def check_relu_fused(session: Session) -> str:
    float_ops = [type(op).__name__ for op in session.compiled.float_graph.ops]
    int8_ops = [type(op).__name__ for op in session.compiled.graph.ops]
    _expect(any("ReLU" in name for name in float_ops), "the FP32 graph has no ReLU")
    standalone = sum("ReLU" in name for name in int8_ops)
    _expect(standalone == 0, f"{standalone} ReLU steps were not folded")
    return f"fp32_ops={len(float_ops)} int8_ops={len(int8_ops)} standalone_relu={standalone}"


def _constant_arrays(session: Session) -> list[tuple[str, str]]:
    text = session.compiled.artifacts.weights_source.read_text()
    return re.findall(r"^(?:_Alignas\(\d+\) )?(const )?(\w+) \w+\[\d+\] = ", text, flags=re.M)


def check_constant_types(session: Session) -> str:
    types = [kind for _, kind in _constant_arrays(session)]
    other = [kind for kind in types if kind not in ("int8_t", "int32_t")]
    _expect(bool(types) and not other, f"unexpected constant types: {other}")
    return f"int8_arrays={types.count('int8_t')} int32_arrays={types.count('int32_t')} other_arrays={len(other)}"


def check_constants_read_only(session: Session) -> str:
    arrays = _constant_arrays(session)
    read_only = sum(1 for qualifier, _ in arrays if qualifier)
    _expect(bool(arrays) and read_only == len(arrays), "a constant array is not declared const")
    return f"const_arrays={read_only}/{len(arrays)}"


def check_calibration_samples(session: Session) -> str:
    count = session.compiled.calibration_report.sample_count
    _expect(count == len(session.calibration), f"report lists {count} samples")
    return f"calibration_samples={count}"


def check_calibration_ranges(session: Session) -> str:
    edges = session.compiled.calibration_report.edges
    valid = [
        edge for edge in edges
        if edge.minimum <= edge.maximum and edge.scale > 0.0 and -128 <= edge.zero_point <= 127
    ]
    _expect(bool(edges) and len(valid) == len(edges), "a calibrated tensor has an invalid range")
    return f"calibrated_tensors={len(edges)} valid_ranges={len(valid)}/{len(edges)}"


def check_deterministic(session: Session) -> str:
    def digests(directory: Path) -> dict[str, str]:
        return {
            path.name: _digest(path)
            for path in sorted(directory.iterdir())
            if path.is_file() and not path.name.startswith(".")
        }

    first = digests(session.compiled.artifacts.output_dir)
    second = digests(session.compile("repeat").artifacts.output_dir)
    different = sorted(name for name in first if first[name] != second.get(name))
    _expect(first.keys() == second.keys() and not different, f"files differ between runs: {different}")
    return f"identical_files={len(first)}/{len(first)}"


def check_recompile_in_place(session: Session) -> str:
    session.backend("in_place")
    artifacts = session.backend("in_place").artifacts
    files = len(bakenn.load_manifest(artifacts.manifest)["artifact_inventory"]["files"])
    return f"replaced_earlier_artifacts=1 hashed_files_verified={files}"


def check_symbol_name(session: Session) -> str:
    model, example, calibration = _tiny(session)
    artifacts = session.once(
        "named",
        lambda: bakenn.compile_torch_ptq(model, example, calibration, session.output / "named", name="my-model"),
    ).artifacts
    symbol = artifacts.model_source.stem
    _expect(re.fullmatch(r"[A-Za-z_]\w*", symbol) is not None, f"{symbol} is not a C identifier")
    _expect(f"void {symbol}_infer(" in artifacts.header.read_text(), "the header lacks the renamed function")
    return f"name=my-model function={symbol}_infer"


# --- Generated files ---------------------------------------------------------


def check_file(attribute: str, session: Session) -> str:
    path = getattr(session.compiled.artifacts, attribute)
    _expect(path.is_file() and path.stat().st_size > 0, f"{path.name} is missing or empty")
    return f"file={path.name} present=1"


def check_no_other_files(session: Session) -> str:
    artifacts = session.compiled.artifacts
    names = {getattr(artifacts, attribute).name for attribute, _ in GENERATED_FILES}
    unexpected = sorted(
        path.name
        for path in artifacts.output_dir.iterdir()
        if path.name not in names and not path.name.startswith(".")
    )
    _expect(not unexpected, f"unexpected generated files: {unexpected}")
    return f"expected_files={len(names)} unexpected_files={len(unexpected)}"


def check_prototype(session: Session) -> str:
    header = " ".join(session.compiled.artifacts.header.read_text().split())
    prototype = (
        "void bknn_mnist_infer( uint8_t *BKNN_RESTRICT arena, "
        "const int8_t *BKNN_RESTRICT input, int8_t *BKNN_RESTRICT output);"
    )
    _expect(prototype in header, "the header does not declare the inference function")
    return "function=bknn_mnist_infer parameters=arena,input,output"


def check_size_macros(session: Session) -> str:
    macros = _header_macros(session.compiled.artifacts)
    plan = session.compiled.plan
    sizes = {
        "INPUT_BYTES": int(np.prod(plan.tensors[plan.inputs[0]].tensor_type.shape)),
        "OUTPUT_BYTES": int(np.prod(plan.tensors[plan.outputs[0]].tensor_type.shape)),
    }
    for name, value in sizes.items():
        _expect(macros[name] == f"{value}u", f"{name} is {macros[name]}, the plan needs {value}")
    return " ".join(f"{name}={value}" for name, value in sizes.items())


def check_shape_macros(session: Session) -> str:
    macros = _header_macros(session.compiled.artifacts)
    plan = session.compiled.plan
    evidence = []
    for side, tensor in (("INPUT", plan.inputs[0]), ("OUTPUT", plan.outputs[0])):
        shape = tuple(plan.tensors[tensor].tensor_type.shape)
        _expect(macros[f"{side}_RANK"] == f"{len(shape)}u", f"{side}_RANK differs from the plan")
        dims = tuple(int(macros[f"{side}_DIM_{axis}"].rstrip("u")) for axis in range(len(shape)))
        _expect(dims == shape, f"{side} dimensions {dims} differ from the plan {shape}")
        evidence.append(f"{side}_DIMS={'x'.join(map(str, dims))}")
    return " ".join(evidence)


def check_qparam_macros(session: Session) -> str:
    macros = _header_macros(session.compiled.artifacts)
    plan = session.compiled.plan
    evidence = []
    for side, tensor in (("INPUT", plan.inputs[0]), ("OUTPUT", plan.outputs[0])):
        qparams = plan.tensors[tensor].tensor_type.qparams
        scale = float.fromhex(macros[f"{side}_SCALE"].rstrip("f"))
        _expect(scale == float(np.float32(qparams.scale)), f"{side}_SCALE differs from the plan")
        _expect(int(macros[f"{side}_ZERO_POINT"]) == qparams.zero_point, f"{side}_ZERO_POINT differs")
        evidence.append(f"{side}_ZERO_POINT={qparams.zero_point}")
    return " ".join(evidence) + " scales_match_plan=2/2"


def check_cmake_fragment(session: Session) -> str:
    artifacts = session.compiled.artifacts
    text = artifacts.build_fragment.read_text()
    listed = [path.name for path in _sources(artifacts) if f'/{path.name}"' in text]
    _expect(len(listed) == 3, f"the CMake fragment lists only {listed}")
    return f"listed_sources={len(listed)}/3"


def check_cpp_header(session: Session) -> str:
    compiler = _require("c++")
    artifacts = session.compiled.artifacts
    directory = session.output / "host" / "cpp_header"
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "include.cpp"
    source.write_text(
        f'#include "{artifacts.header.name}"\n'
        "int main() { return BKNN_MNIST_OUTPUT_BYTES == 10u ? 0 : 1; }\n"
    )
    subprocess.run(
        [compiler, "-std=c++11", *WARNINGS, "-fsyntax-only", "-I", str(artifacts.output_dir), str(source)],
        check=True,
    )
    _expect('extern "C"' in artifacts.header.read_text(), 'the header has no extern "C" guard')
    return "cpp_standard=c++11 compiles=1 extern_c_guard=1"


# --- Properties of the generated C -------------------------------------------


def _generated_c(session: Session) -> str:
    return "\n".join(path.read_text() for path in _sources(session.compiled.artifacts))


def check_no_heap(session: Session) -> str:
    calls = re.findall(r"\b(?:malloc|calloc|realloc|free)\s*\(", _generated_c(session))
    _expect(not calls, f"generated C calls the heap: {calls}")
    return f"heap_calls={len(calls)}"


def check_no_float(session: Session) -> str:
    uses = re.findall(r"\b(?:float|double)\b", _generated_c(session))
    _expect(not uses, f"generated C uses floating point {len(uses)} times")
    return f"float_types={len(uses)}"


def check_includes(session: Session) -> str:
    artifacts = session.compiled.artifacts
    files = (*_sources(artifacts), artifacts.header, artifacts.weights_header, artifacts.kernels_header)
    headers = sorted({name for path in files for name in re.findall(r"#include <([^>]+)>", path.read_text())})
    other = [name for name in headers if name not in ("limits.h", "stddef.h", "stdint.h")]
    _expect(bool(headers) and not other, f"generated C includes {other}")
    return f"system_headers={','.join(headers)} other_headers={len(other)}"


def check_strict_c(standard: str, session: Session) -> str:
    artifacts = session.compiled.artifacts
    for source in _sources(artifacts):
        subprocess.run(
            [
                session.compiler, f"-std={standard}", "-pedantic-errors", "-Wall", "-Wextra", "-Werror",
                "-fsyntax-only", "-I", str(artifacts.output_dir), str(source),
            ],
            check=True,
        )
    return f"standard={standard} sources=3 diagnostics=0"


def check_optimization(level: str, session: Session) -> str:
    return f"optimization={level} " + _mismatched(session, f"mnist{level}", level)


def check_sanitizers(session: Session) -> str:
    def build() -> tuple[np.ndarray, str]:
        _, inputs, expected, _ = _corpus(session)
        flags = ("-std=c11", "-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=undefined")
        try:
            executable = _build(session, session.compiled.artifacts, "mnist_sanitized", *flags)
        except subprocess.CalledProcessError as error:
            raise Skip("the compiler has no AddressSanitizer runtime") from error
        completed = subprocess.run(
            [str(executable)], input=inputs.tobytes(), capture_output=True,
            env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0"},
        )
        _expect(completed.returncode == 0, f"a sanitizer stopped the run: {completed.stderr.decode()[:300]}")
        return np.frombuffer(completed.stdout, dtype=np.int8).reshape(expected.shape), completed.stderr.decode()

    actual, report = session.once("host.sanitized", build)
    _expect("runtime error" not in report and "ERROR" not in report, f"sanitizer report: {report[:300]}")
    mismatched = int(np.count_nonzero(actual != _corpus(session)[2]))
    _expect(mismatched == 0, f"{mismatched} bytes differ under the sanitizers")
    return f"sanitizer_reports=0 mismatched_bytes={mismatched}/{actual.size}"


def _guard_counts(session: Session) -> tuple[int, int, int, int]:
    def build() -> tuple[int, int, int, int]:
        artifacts = session.compiled.artifacts
        executable = _build(session, artifacts, "mnist_guarded", "-std=c11", "-O2", source=_guard_source(artifacts))
        completed = subprocess.run(
            [str(executable)], input=_corpus(session)[1].tobytes(), capture_output=True, check=True
        )
        images, guards, inputs, repeats = map(int, completed.stdout.split())
        return images, guards, inputs, repeats

    return session.once("host.guarded", build)


def check_guards(session: Session) -> str:
    images, damaged, _, _ = _guard_counts(session)
    _expect(images == 100 and damaged == 0, f"{damaged} guard bytes were overwritten")
    return f"images={images} guard_bytes_overwritten={damaged}"


def check_input_unchanged(session: Session) -> str:
    images, _, changed, _ = _guard_counts(session)
    _expect(images == 100 and changed == 0, f"{changed} input buffers were modified")
    return f"images={images} inputs_modified={changed}"


def check_repeatable(session: Session) -> str:
    images, _, _, different = _guard_counts(session)
    _expect(images == 100 and different == 0, f"{different} repeated runs gave another output")
    return f"images={images} repeated_runs_differing={different}"


# --- Output verification -----------------------------------------------------


def check_frozen_outputs(session: Session) -> str:
    return _mismatched(session, "mnist-O2", "-O2")


def check_classification(session: Session) -> str:
    labels = _corpus(session)[3]
    correct = int(np.count_nonzero(np.argmax(_mnist_outputs(session, "mnist-O2", "-O2"), axis=1) == labels))
    _expect(correct >= 95, f"only {correct} of {len(labels)} images are classified correctly")
    return f"correct={correct}/{len(labels)}"


def check_fp32_parity(session: Session) -> str:
    pixels, _, _, labels = _corpus(session)
    with torch.no_grad():
        logits = session.mnist()(torch.from_numpy(pixels.copy()).unsqueeze(1).float() / 255.0)
    fp32 = int(np.count_nonzero(logits.argmax(dim=1).numpy() == labels))
    int8 = int(np.count_nonzero(np.argmax(_mnist_outputs(session, "mnist-O2", "-O2"), axis=1) == labels))
    _expect(abs(fp32 - int8) <= 1, f"FP32 classifies {fp32} images correctly, INT8 {int8}")
    return f"fp32_correct={fp32}/{len(labels)} int8_correct={int8}/{len(labels)}"


def check_reference(session: Session) -> str:
    inputs = _corpus(session)[1]
    actual = _mnist_outputs(session, "mnist-O2", "-O2")[:10]
    reference = np.concatenate(
        [bakenn.run_reference(session.compiled.plan, inputs[index : index + 1]) for index in range(10)]
    ).reshape(10, -1)
    mismatched = int(np.count_nonzero(actual != reference))
    _expect(mismatched == 0, f"{mismatched} bytes differ from the Python reference")
    return f"python_reference_mismatched_bytes={mismatched}/{reference.size}"


def check_quantize_input(session: Session) -> str:
    pixels = _corpus(session)[0]
    frozen = np.fromfile(EVIDENCE / "physical_test_inputs_int8.bin", dtype=np.int8)
    values = (pixels.astype(np.float32) / 255.0)[..., None]
    quantized = np.concatenate(
        [bakenn.quantize_input(session.compiled.plan, values[index : index + 1]) for index in range(len(values))]
    )
    mismatched = int(np.count_nonzero(quantized.reshape(-1) != frozen))
    _expect(quantized.dtype == np.int8 and mismatched == 0, f"{mismatched} input codes differ")
    return f"mismatched_bytes={mismatched}/{frozen.size}"


def check_dequantize_output(session: Session) -> str:
    plan = session.compiled.plan
    qparams = plan.tensors[plan.outputs[0]].tensor_type.qparams
    codes = _corpus(session)[2][:1]
    values = bakenn.dequantize_output(plan, codes)
    expected = (codes.astype(np.float32) - qparams.zero_point) * np.float32(qparams.scale)
    difference = float(np.abs(values - expected).max())
    _expect(difference < 1e-6, f"dequantized values differ by {difference}")
    return f"values={values.size} max_abs_difference={difference:.6f}"


def check_trace(session: Session) -> str:
    plan = session.compiled.plan
    sample = _corpus(session)[1][:1]
    trace = bakenn.run_reference_trace(plan, sample)
    _expect(set(plan.inputs) | set(plan.outputs) <= set(trace), "the trace lacks the model input or output")
    final = np.array_equal(trace[plan.outputs[0]], bakenn.run_reference(plan, sample))
    _expect(final, "the traced output differs from run_reference")
    return f"traced_tensors={len(trace)} output_matches_run_reference={int(final)}"


# --- Accuracy report ---------------------------------------------------------


def _accuracy(session: Session):  # type: ignore[no-untyped-def]
    return session.once("accuracy", lambda: session.compiled.verify_accuracy(session.calibration[:20]))


def check_accuracy_samples(session: Session) -> str:
    count = _accuracy(session).sample_count
    _expect(count == 20, f"the report used {count} samples")
    return f"samples={count}"


def check_accuracy_layers(session: Session) -> str:
    layers = _accuracy(session).layers
    _expect(len(layers) > 1 and all(layer.element_count > 0 for layer in layers), "a layer row is empty")
    return f"layers={len(layers)}"


def check_accuracy_errors(session: Session) -> str:
    output = _accuracy(session).layers[-1]
    ordered = 0.0 <= output.mean_absolute_error <= output.root_mean_square_error <= output.maximum_absolute_error
    _expect(ordered and np.isfinite(output.maximum_absolute_error), "the output error statistics are inconsistent")
    return (
        f"output_max_abs_error={output.maximum_absolute_error:.4f} "
        f"output_mean_abs_error={output.mean_absolute_error:.4f} "
        f"output_rms_error={output.root_mean_square_error:.4f}"
    )


def check_accuracy_endpoints(session: Session) -> str:
    layers = _accuracy(session).layers
    valid = [layer for layer in layers if 0 <= layer.int8_endpoint_count <= layer.element_count]
    _expect(len(valid) == len(layers), "an endpoint count exceeds its element count")
    return f"layers_with_endpoint_count={len(valid)}/{len(layers)}"


def check_accuracy_max_samples(session: Session) -> str:
    count = session.compiled.verify_accuracy(session.calibration[:20], max_samples=5).sample_count
    _expect(count == 5, f"max_samples=5 used {count} samples")
    return f"given_samples=20 max_samples=5 used_samples={count}"


# --- Memory report and budgets -----------------------------------------------


def check_arena_report(session: Session) -> str:
    arena = session.compiled.memory_report.arena_bytes
    _expect(arena > 0, "the report lists no arena")
    return f"arena_bytes={arena}"


def check_arena_macro(session: Session) -> str:
    arena = session.compiled.memory_report.arena_bytes
    macro = _header_macros(session.compiled.artifacts)["ARENA_SIZE"]
    _expect(macro == f"{arena}u", f"the header says {macro}, the report {arena}")
    return f"header_macro={macro.rstrip('u')} report={arena}"


def check_arena_manifest(session: Session) -> str:
    arena = session.compiled.memory_report.arena_bytes
    manifest = _manifest(session.compiled.artifacts)["arena_bytes"]
    _expect(manifest == arena, f"the manifest says {manifest}, the report {arena}")
    return f"manifest={manifest} report={arena}"


def check_arena_alignment(session: Session) -> str:
    alignment = session.compiled.memory_report.arena_alignment
    macro = _header_macros(session.compiled.artifacts)["ARENA_ALIGNMENT"]
    _expect(macro == f"{alignment}u", f"the header says {macro}, the report {alignment}")
    return f"arena_alignment={alignment} header_macro={macro.rstrip('u')}"


def check_constant_bytes(session: Session) -> str:
    report = session.compiled.memory_report.emitted_constant_payload_bytes
    manifest = _manifest(session.compiled.artifacts)["constant_bytes"]
    _expect(report == manifest and report > 0, f"the manifest says {manifest}, the report {report}")
    return f"constant_bytes={report} manifest={manifest}"


def check_caller_io(session: Session) -> str:
    report = session.compiled.memory_report
    _expect(report.caller_io_bytes == report.input_bytes + report.output_bytes, "caller I/O does not add up")
    return f"input_bytes={report.input_bytes} output_bytes={report.output_bytes} caller_io_bytes={report.caller_io_bytes}"


def check_heap_report(session: Session) -> str:
    calls = session.compiled.memory_report.generated_model_heap_calls
    _expect(calls == 0, f"the report lists {calls} heap calls")
    return f"generated_model_heap_calls={calls}"


def check_buffers_in_arena(session: Session) -> str:
    report = session.compiled.memory_report
    inside = [item for item in report.buffers if 0 <= item.offset and item.offset + item.size_bytes <= report.arena_bytes]
    _expect(bool(report.buffers) and len(inside) == len(report.buffers), "a buffer leaves the arena")
    return f"buffers={len(report.buffers)} inside_arena={len(inside)}/{len(report.buffers)}"


def check_arena_reuse(session: Session) -> str:
    report = session.compiled.memory_report
    total = sum(item.size_bytes for item in report.buffers)
    _expect(bool(report.reuse_regions) and total > report.arena_bytes, "no arena memory is reused")
    return f"buffer_bytes_total={total} arena_bytes={report.arena_bytes} reused_regions={len(report.reuse_regions)}"


def check_peak(session: Session) -> str:
    report = session.compiled.memory_report
    _expect(0 < report.peak_working_payload_bytes <= report.arena_bytes, "the peak exceeds the arena")
    return f"peak_working_bytes={report.peak_working_payload_bytes} arena_bytes={report.arena_bytes}"


def check_memory_text(session: Session) -> str:
    artifacts = session.compiled.artifacts
    arena = session.compiled.memory_report.arena_bytes
    _expect(f"({arena} B)" in artifacts.memory_report_text.read_text(), "the text report lacks the arena size")
    return f"file={artifacts.memory_report_text.name} lists_arena_bytes=1"


def check_memory_json(session: Session) -> str:
    artifacts = session.compiled.artifacts
    arena = json.loads(artifacts.memory_report_json.read_text())["compile_time"]["arena_bytes"]
    _expect(arena == session.compiled.memory_report.arena_bytes, "the JSON report lists another arena size")
    return f"file={artifacts.memory_report_json.name} arena_bytes={arena}"


def _budget(session: Session, name: str, **limits: int):  # type: ignore[no-untyped-def]
    return session.backend(f"budget/{name}", target=replace(bakenn.PORTABLE_32, **limits))


def check_sram_rejected(session: Session) -> str:
    arena = session.compiled.memory_report.arena_bytes
    budget = arena // 2
    _rejection(lambda: _budget(session, "sram_small", sram_bytes=budget), f"exceeds SRAM budget {budget}")
    return f"arena_bytes={arena} sram_budget={budget} rejected=1"


def check_sram_nothing_written(session: Session) -> str:
    budget = session.compiled.memory_report.arena_bytes - 1
    _rejection(lambda: _budget(session, "sram_short", sram_bytes=budget), "exceeds SRAM budget")
    written = int((session.output / "budget/sram_short").exists())
    _expect(written == 0, "a rejected compile left artifacts behind")
    return f"sram_budget={budget} artifacts_written={written}"


def check_sram_accepted(session: Session) -> str:
    arena = session.compiled.memory_report.arena_bytes
    session.once("budget.sram_exact", lambda: _budget(session, "sram_exact", sram_bytes=arena))
    return f"arena_bytes={arena} sram_budget={arena} accepted=1"


def check_flash_rejected(session: Session) -> str:
    constants = session.compiled.memory_report.emitted_constant_payload_bytes
    budget = constants // 2
    _rejection(lambda: _budget(session, "flash_small", flash_bytes=budget), f"exceeds Flash budget {budget}")
    return f"constant_bytes={constants} flash_budget={budget} rejected=1"


def check_flash_nothing_written(session: Session) -> str:
    budget = session.compiled.memory_report.emitted_constant_payload_bytes - 1
    _rejection(lambda: _budget(session, "flash_short", flash_bytes=budget), "exceeds Flash budget")
    written = int((session.output / "budget/flash_short").exists())
    _expect(written == 0, "a rejected compile left artifacts behind")
    return f"flash_budget={budget} artifacts_written={written}"


def check_flash_accepted(session: Session) -> str:
    constants = session.compiled.memory_report.emitted_constant_payload_bytes
    session.once("budget.flash_exact", lambda: _budget(session, "flash_exact", flash_bytes=constants))
    return f"constant_bytes={constants} flash_budget={constants} accepted=1"


# --- Error handling ----------------------------------------------------------


def _tanh_error(session: Session) -> Exception:
    def build() -> Exception:
        model = _seeded(session, lambda: nn.Sequential(nn.Linear(8, 4), nn.Tanh()), 3)
        sample = torch.zeros(1, 8)
        return _rejection(
            lambda: bakenn.compile_torch_ptq(model, sample, sample, session.output / "rejected/tanh"),
            "unsupported torch.export operator aten.tanh.default",
        )

    return session.once("tanh", build)


def check_unsupported_operator(session: Session) -> str:
    _tanh_error(session)
    return 'operator=tanh rejected=1 message_contains="unsupported torch.export operator"'


def check_unsupported_diagnostic(session: Session) -> str:
    error = _tanh_error(session)
    _expect(error.code == "BAKENN_TORCH_OPERATOR_UNSUPPORTED", f"unexpected code {error.code}")  # type: ignore[attr-defined]
    _expect(len(error.suggestions) > 0, "the error suggests no next step")  # type: ignore[attr-defined]
    return f"code={error.code} suggestions={len(error.suggestions)}"  # type: ignore[attr-defined]


def check_unsupported_nothing_written(session: Session) -> str:
    _tanh_error(session)
    written = int((session.output / "rejected/tanh").exists())
    _expect(written == 0, "a rejected compile left artifacts behind")
    return f"artifacts_written={written}"


def check_training_mode(session: Session) -> str:
    model = _seeded(session, lambda: nn.Sequential(nn.Conv2d(1, 4, 3), nn.BatchNorm2d(4)), 5).train()
    return _rejected_compile(session, "training", "requires eval mode", model=model)


def check_batch_size(session: Session) -> str:
    return _rejected_compile(session, "batch", "static batch-one", example=torch.zeros(2, 1, 8, 8))


def check_example_dtype(session: Session) -> str:
    example = torch.zeros(1, 1, 8, 8, dtype=torch.float64)
    return _rejected_compile(session, "dtype", "example input must be float32", example=example)


def check_example_nan(session: Session) -> str:
    example = torch.full((1, 1, 8, 8), float("nan"))
    return _rejected_compile(session, "example_nan", "example input contains NaN or infinity", example=example)


def check_two_inputs(session: Session) -> str:
    example = (torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 8))
    return _rejected_compile(session, "two_inputs", "exactly one tensor example input", example=example)


def check_empty_calibration(session: Session) -> str:
    return _rejected_compile(session, "empty", "at least one sample", calibration=torch.zeros(0, 1, 8, 8))


def check_calibration_shape(session: Session) -> str:
    calibration = torch.zeros(4, 1, 9, 9)
    return _rejected_compile(session, "shape", "is incompatible with batch-one input", calibration=calibration)


def check_calibration_nan(session: Session) -> str:
    calibration = torch.full((4, 1, 8, 8), float("nan"))
    return _rejected_compile(
        session, "calibration_nan", "calibration data contains NaN or infinity", calibration=calibration
    )


def check_not_a_module(session: Session) -> str:
    return _rejected_compile(session, "not_module", "requires a torch.nn.Module", model="not a model")


def check_output_is_file(session: Session) -> str:
    model, example, calibration = _tiny(session)
    path = session.output / "occupied.txt"
    path.write_text("not a directory\n")
    evidence = _refused(
        lambda: bakenn.compile_torch_ptq(model, example, calibration, path), "exists and is not a directory"
    )
    _expect(path.read_text() == "not a directory\n", "the existing file was modified")
    return evidence


def check_foreign_directory(session: Session) -> str:
    directory = session.output / "foreign"
    directory.mkdir(exist_ok=True)
    (directory / "keep.txt").write_text("someone else's file\n")
    _rejection(
        lambda: bakenn.compile(session.compiled.graph, directory),
        "refusing to replace non-empty unmanaged output directory",
    )
    kept = sorted(path.name for path in directory.iterdir() if not path.name.startswith("."))
    _expect(kept == ["keep.txt"], f"the foreign directory now holds {kept}")
    return "rejected=1 foreign_files_kept=1 files_added=0"


def check_unknown_target(session: Session) -> str:
    error = _rejection(
        lambda: bakenn.compile(session.compiled.graph, session.output / "rejected/target", target="z80"),
        "unknown BakeNN target 'z80'",
        ValueError,
    )
    named = [name for name in PROFILES if name in str(error)]
    _expect(len(named) == len(PROFILES), "the error does not list the valid targets")
    return f"rejected=1 valid_targets_listed={len(named)}/{len(PROFILES)}"


def check_require_optimized(session: Session) -> str:
    options = bakenn.CBackendOptions(kernel_policy=bakenn.KernelPolicy.REQUIRE_OPTIMIZED)
    return _refused(
        lambda: session.backend("rejected/require_optimized", backend_options=options),
        "kernel policy require_optimized has no supported implementation",
    )


def check_dram_needs_esp(session: Session) -> str:
    options = bakenn.CBackendOptions(requantization_in_dram=True)
    return _refused(
        lambda: session.backend("rejected/dram", backend_options=options), "requires an ESP-IDF target"
    )


def check_quantize_shape(session: Session) -> str:
    values = np.zeros((1, 9, 9, 1), dtype=np.float32)
    return _refused(lambda: bakenn.quantize_input(session.compiled.plan, values), "expected (1, 28, 28, 1)")


def check_reference_dtype(session: Session) -> str:
    values = np.zeros((1, 28, 28, 1), dtype=np.float32)
    return _refused(lambda: bakenn.run_reference(session.compiled.plan, values), "must have dtype int8")


def check_reference_shape(session: Session) -> str:
    values = np.zeros((1, 9, 9, 1), dtype=np.int8)
    return _refused(lambda: bakenn.run_reference(session.compiled.plan, values), "expected (1, 28, 28, 1)")


def check_dequantize_shape(session: Session) -> str:
    values = np.zeros((2, 10), dtype=np.int8)
    return _refused(lambda: bakenn.dequantize_output(session.compiled.plan, values), "expected (1, 10)")


def check_manifest_missing(session: Session) -> str:
    return _refused(lambda: bakenn.load_manifest(session.output / "no_such_manifest.json"), "cannot read")


def check_manifest_malformed(session: Session) -> str:
    path = session.output / "malformed_manifest.json"
    path.write_text("{not json")
    return _refused(lambda: bakenn.load_manifest(path), "malformed JSON")


def check_build_target_mismatch(session: Session) -> str:
    _require("arm-none-eabi-gcc")
    return _refused(
        lambda: bakenn.build_freestanding_elf(
            session.compiled.artifacts, bakenn.CORTEX_M4, session.output / "rejected/elf"
        ),
        "does not match build target cortex-m4",
    )


def check_idf_needs_esp(session: Session) -> str:
    return _refused(
        lambda: bakenn.export_esp_idf_project(
            session.compiled.artifacts, bakenn.PORTABLE_32, session.output / "rejected/idf"
        ),
        "is not an ESP-IDF target",
    )


def check_zephyr_needs_cortex_m4(session: Session) -> str:
    return _refused(
        lambda: bakenn.export_zephyr_project(
            session.compiled.artifacts, bakenn.PORTABLE_32, session.output / "rejected/zephyr"
        ),
        "requires the cortex-m4 DSP target",
    )


def check_zephyr_board(session: Session) -> str:
    return _refused(
        lambda: bakenn.export_zephyr_project(
            _cortex_m4(session).artifacts, bakenn.CORTEX_M4, session.output / "rejected/board", board="no_such_board"
        ),
        "unsupported Zephyr IoT-LAB board",
    )


def check_arduino_vendor_kernels(session: Session) -> str:
    return _refused(
        lambda: bakenn.export_arduino_library(_cortex_m4(session).artifacts, session.output / "rejected/arduino"),
        "requires artifacts without bundled CMSIS-NN or ESP-NN sources",
    )


def check_arduino_existing(session: Session) -> str:
    library = _arduino(session)
    before = _digest(library.root / "library.properties")
    evidence = _refused(
        lambda: bakenn.export_arduino_library(session.compiled.artifacts, library.root),
        "refusing to overwrite non-empty export directory",
    )
    _expect(_digest(library.root / "library.properties") == before, "the existing library was modified")
    return evidence


def check_arduino_version(session: Session) -> str:
    return _refused(
        lambda: bakenn.export_arduino_library(
            session.compiled.artifacts, session.output / "rejected/arduino_version", version="one"
        ),
        "expected MAJOR.MINOR.PATCH",
    )


def check_arduino_name(session: Session) -> str:
    return _refused(
        lambda: bakenn.export_arduino_library(
            session.compiled.artifacts, session.output / "rejected/arduino_name", name="bad name!"
        ),
        "invalid Arduino library name",
    )


# --- Artifact integrity ------------------------------------------------------


def _tampered(session: Session, name: str, change: Callable[[Path], None]) -> Path:
    """Copy the generated files, apply one change, and return the copied manifest."""

    artifacts = session.compiled.artifacts
    copy = session.output / "tampered" / name
    shutil.rmtree(copy, ignore_errors=True)
    shutil.copytree(artifacts.output_dir, copy, ignore=shutil.ignore_patterns(".*"))
    change(copy)
    return copy / artifacts.manifest.name


def check_manifest_intact(session: Session) -> str:
    files = bakenn.load_manifest(session.compiled.artifacts.manifest)["artifact_inventory"]["files"]
    return f"hashed_files_verified={len(files)}"


def check_bit_flip(session: Session) -> str:
    name = session.compiled.artifacts.weights_source.name

    def flip(copy: Path) -> None:
        data = bytearray((copy / name).read_bytes())
        data[len(data) // 2] ^= 0x01
        (copy / name).write_bytes(bytes(data))

    manifest = _tampered(session, "bit_flip", flip)
    return _refused(lambda: bakenn.load_manifest(manifest), "SHA-256 does not match inventory")


def check_truncated_file(session: Session) -> str:
    name = session.compiled.artifacts.kernels_source.name

    def truncate(copy: Path) -> None:
        (copy / name).write_bytes((copy / name).read_bytes()[:-1])

    manifest = _tampered(session, "truncated", truncate)
    return _refused(lambda: bakenn.load_manifest(manifest), "byte count does not match inventory")


def check_missing_file(session: Session) -> str:
    name = session.compiled.artifacts.memory_report_text.name
    manifest = _tampered(session, "missing", lambda copy: (copy / name).unlink())
    return _refused(lambda: bakenn.load_manifest(manifest), f"missing=['{name}']")


def check_added_file(session: Session) -> str:
    manifest = _tampered(session, "added", lambda copy: (copy / "extra.c").write_text("int extra;\n"))
    return _refused(lambda: bakenn.load_manifest(manifest), "unexpected=['extra.c']")


def check_edited_manifest(session: Session) -> str:
    name = session.compiled.artifacts.manifest.name
    arena = session.compiled.memory_report.arena_bytes

    def edit(copy: Path) -> None:
        text = (copy / name).read_text()
        _expect(f'"arena_bytes": {arena}' in text, "the manifest does not record the arena size")
        (copy / name).write_text(text.replace(f'"arena_bytes": {arena}', f'"arena_bytes": {arena + 1}', 1))

    manifest = _tampered(session, "edited", edit)
    return _refused(lambda: bakenn.load_manifest(manifest), "invalid BakeNN manifest")


def check_manifest_identity(session: Session) -> str:
    manifest = _manifest(session.compiled.artifacts)
    target = manifest["backend"]["target"]["id"]
    kernels = [item["implementation"] for item in manifest["backend"]["selections"]]
    _expect(manifest["model"] == "bknn_mnist" and target == "portable32", "unexpected model or target")
    _expect(kernels == _kernels(session.compiled.artifacts), "the manifest lists other kernels")
    return f"model={manifest['model']} target={target} kernels={len(kernels)}"


def check_manifest_rejections(session: Session) -> str:
    selections = _manifest(_cortex_m4(session).artifacts)["backend"]["selections"]
    reasons = [reason for item in selections for reason in item["rejected_implementations"].values()]
    _expect(bool(reasons) and all(reasons), "a rejected kernel has no recorded reason")
    with_candidates = sum(1 for item in selections if item["rejected_implementations"])
    return f"steps_with_alternatives={with_candidates}/{len(selections)} recorded_reasons={len(reasons)}"


def check_fingerprints(session: Session) -> str:
    fingerprints = _manifest(session.compiled.artifacts)["graph_fingerprints"]
    names = ("canonical_graph_sha256", "execution_plan_sha256", "semantic_constants_sha256")
    for name in names:
        _expect(re.fullmatch(r"[0-9a-f]{64}", fingerprints[name]) is not None, f"{name} is not a SHA-256")
    return f"sha256_fingerprints={len(names)}/{len(names)}"


def check_arithmetic_profile(session: Session) -> str:
    profile = _manifest(session.compiled.artifacts)["arithmetic_profile"]
    header = session.compiled.artifacts.header.read_text()
    _expect(f'#define BKNN_ARITHMETIC_PROFILE_ID "{profile}"' in header, "the header names another profile")
    return f"arithmetic_profile={profile} header_macro_matches=1"


def _frozen(session: Session) -> dict:
    def build() -> dict:
        from verify_mnist_evidence import verify

        return verify(EVIDENCE, session.compiler)

    return session.once("frozen", build)


def check_frozen_hashes(session: Session) -> str:
    files = _frozen(session)["verified_payload_files"]
    _expect(files > 0, "no frozen file was verified")
    return f"frozen_files_verified={files}"


def check_frozen_rebuild(session: Session) -> str:
    result = _frozen(session)
    _expect(result["mismatched_output_bytes"] == 0, "the frozen generated C gave other outputs")
    return f"compared_bytes={result['compared_output_bytes']} mismatched_output_bytes={result['mismatched_output_bytes']}"


# --- Portable targets --------------------------------------------------------


def check_portable_kernels(session: Session) -> str:
    kernels = _kernels(session.compiled.artifacts)
    portable = [kernel for kernel in kernels if kernel.startswith("portable.")]
    _expect(len(portable) == len(kernels), f"non-portable kernels: {set(kernels) - set(portable)}")
    return f"portable_kernels={len(portable)}/{len(kernels)}"


def _core(session: Session, cpu: str):  # type: ignore[no-untyped-def]
    """Generate and link the MNIST model for one Arm core; return the report and a source digest."""

    def build():  # type: ignore[no-untyped-def]
        _require("arm-none-eabi-gcc")
        target = bakenn.TargetDescriptor(
            target_id=f"any-{cpu}",
            architecture=bakenn.TargetArchitecture.ARM,
            cpu=cpu,
            abi="aapcs32-soft",
            toolchain="arm-none-eabi",
            features=frozenset({"scalar-int8", "thumb"}),
            arena_alignment=8,
            constant_alignment=4,
            compiler_flags=(f"-mcpu={cpu}", "-mthumb", "-mfloat-abi=soft"),
        )
        built = session.backend(f"cores/{cpu}", target=target).artifacts
        report = bakenn.build_freestanding_elf(built, target, session.output / f"cores/{cpu}_elf")
        return report, _digest(built.header, *_sources(built))

    return session.once(f"core.{cpu}", build)


def check_core(cpu: str, session: Session) -> str:
    report, _ = _core(session, cpu)
    _expect(not report.undefined_symbols, f"{cpu}: undefined symbols {report.undefined_symbols}")
    _expect(not report.forbidden_symbols, f"{cpu}: heap or float symbols {report.forbidden_symbols}")
    return f"cpu={cpu} undefined_symbols=0 heap_or_float_symbols=0"


def check_identical_sources(session: Session) -> str:
    digests = {_core(session, cpu)[1] for cpu in ARM_CORES}
    _expect(len(digests) == 1, "the generated C differs between cores")
    return f"cores={len(ARM_CORES)} distinct_c_sources={len(digests)}"


def check_image_size(session: Session) -> str:
    report, _ = _core(session, ARM_CORES[0])
    constants = session.compiled.memory_report.emitted_constant_payload_bytes
    _expect(report.flash_load_bytes > constants, "the Flash image is smaller than the model constants")
    _expect(report.static_sram_bytes >= report.model_arena_bytes > 0, "static SRAM is smaller than the arena")
    return (
        f"cpu={ARM_CORES[0]} flash_bytes={report.flash_load_bytes} "
        f"static_sram_bytes={report.static_sram_bytes} model_arena_bytes={report.model_arena_bytes}"
    )


def check_profile(name: str, session: Session) -> str:
    artifacts = session.once(f"profile.{name}", lambda: session.backend(f"profiles/{name}", target=name)).artifacts
    target = _manifest(artifacts)["backend"]["target"]
    kernels = _kernels(artifacts)
    _expect(target["id"] == name, f"the manifest names target {target['id']}")
    _expect(all(kernel.startswith("portable.") for kernel in kernels), f"non-portable kernels: {kernels}")
    return f"target={target['id']} cpu={target['cpu']} portable_kernels={len(kernels)}/{len(kernels)}"


def check_riscv_link(session: Session) -> str:
    def build():  # type: ignore[no-untyped-def]
        _require(*RISCV_COMPILERS)
        built = session.backend("riscv/rv32imc", target="rv32imc").artifacts
        return bakenn.build_freestanding_elf(built, "rv32imc", session.output / "riscv/rv32imc_elf")

    report = session.once("riscv", build)
    _expect(not report.undefined_symbols, f"undefined symbols: {report.undefined_symbols}")
    _expect(not report.forbidden_symbols, f"heap or float symbols: {report.forbidden_symbols}")
    return "target=rv32imc undefined_symbols=0 heap_or_float_symbols=0"


def check_vendor_fallback(session: Session) -> str:
    options = bakenn.CBackendOptions(kernel_policy=bakenn.KernelPolicy.STATIC_PRIORITY, enable_cmsis_nn=True)
    compiled = session.once(
        "cmsis_without_dsp", lambda: session.backend("cmsis_without_dsp", backend_options=options)
    )
    kernels = _kernels(compiled.artifacts)
    vendor = [kernel for kernel in kernels if kernel.startswith("cmsis_nn.")]
    _expect(not vendor, f"CMSIS-NN kernels were selected without DSP: {vendor}")
    return f"cmsis_nn_requested=1 target=portable32 cmsis_nn_kernels={len(vendor)}/{len(kernels)}"


# --- Cortex-M4 and CMSIS-NN --------------------------------------------------


def _cortex_m4(session: Session, policy: bakenn.KernelPolicy = bakenn.KernelPolicy.STATIC_PRIORITY):  # type: ignore[no-untyped-def]
    target = bakenn.CORTEX_M4
    options = bakenn.CBackendOptions(kernel_policy=policy, enable_cmsis_nn=True, target=target)
    return session.once(
        f"cortex_m4.{policy.value}",
        lambda: session.backend(f"cortex_m4/{policy.value}", backend_options=options, target=target),
    )


def _cortex_m4_elf(session: Session):  # type: ignore[no-untyped-def]
    def build():  # type: ignore[no-untyped-def]
        _require("arm-none-eabi-gcc")
        return bakenn.build_freestanding_elf(
            _cortex_m4(session).artifacts, bakenn.CORTEX_M4, session.output / "cortex_m4/elf"
        )

    return session.once("cortex_m4.elf", build)


def check_cmsis_selected(session: Session) -> str:
    kernels = _kernels(_cortex_m4(session).artifacts)
    vendor = [kernel for kernel in kernels if kernel.startswith("cmsis_nn.")]
    _expect(bool(vendor), f"no CMSIS-NN kernel was selected: {kernels}")
    return f"cmsis_nn_kernels={len(vendor)}/{len(kernels)}"


def check_dsp_linear(session: Session) -> str:
    kernels = _kernels(_cortex_m4(session).artifacts)
    _expect("cortex_m4.linear_smlad.v1" in kernels, f"the DSP linear kernel was not selected: {kernels}")
    return "step=linear kernel=cortex_m4.linear_smlad.v1"


def check_portable_step(session: Session) -> str:
    kernels = _kernels(_cortex_m4(session).artifacts)
    _expect("portable.flatten_view.v1" in kernels, f"flatten did not keep its portable kernel: {kernels}")
    return "step=flatten kernel=portable.flatten_view.v1"


def check_cmsis_sources(session: Session) -> str:
    artifacts = _cortex_m4(session).artifacts
    sources = [path for path in artifacts.support_sources if path.is_file()]
    _expect(bool(sources) and len(sources) == len(artifacts.support_sources), "a bundled source is missing")
    return f"bundled_cmsis_nn_sources={len(sources)}"


def check_cmsis_licenses(session: Session) -> str:
    artifacts = _cortex_m4(session).artifacts
    licenses = sorted(path.parent.name for path in artifacts.third_party_licenses if path.is_file())
    _expect(licenses == ["cmsis_core", "cmsis_nn"], f"bundled licenses: {licenses}")
    return f"bundled_licenses={','.join(licenses)}"


def check_cmsis_manifest(session: Session) -> str:
    files = bakenn.load_manifest(_cortex_m4(session).artifacts.manifest)["artifact_inventory"]["files"]
    return f"hashed_files_verified={len(files)}"


def check_cortex_m4_undefined(session: Session) -> str:
    report = _cortex_m4_elf(session)
    _expect(not report.undefined_symbols, f"undefined symbols: {report.undefined_symbols}")
    return f"cpu=cortex-m4 undefined_symbols={len(report.undefined_symbols)}"


def check_cortex_m4_forbidden(session: Session) -> str:
    report = _cortex_m4_elf(session)
    _expect(not report.forbidden_symbols, f"heap or float symbols: {report.forbidden_symbols}")
    return f"cpu=cortex-m4 heap_or_float_symbols={len(report.forbidden_symbols)}"


def check_policy_measured(session: Session) -> str:
    kernels = _kernels(_cortex_m4(session, bakenn.KernelPolicy.MEASURED).artifacts)
    portable = [kernel for kernel in kernels if kernel.startswith("portable.")]
    _expect(len(portable) == len(kernels), "MEASURED chose a kernel without a measurement")
    return f"policy=measured measurements=0 portable_kernels={len(portable)}/{len(kernels)}"


# --- ESP32, ESP-NN and ESP-IDF -----------------------------------------------


def _esp(session: Session, target: bakenn.TargetDescriptor, **options: bool):  # type: ignore[no-untyped-def]
    name = target.target_id + "".join(f"_{key}" for key in sorted(options))

    def build():  # type: ignore[no-untyped-def]
        from generate_smoke import esp_nn_smoke_graph

        return bakenn.compile(
            esp_nn_smoke_graph(),
            session.output / "esp" / name,
            backend_options=bakenn.CBackendOptions(
                kernel_policy=bakenn.KernelPolicy.STATIC_PRIORITY, enable_esp_nn=True, target=target, **options
            ),
            target=target,
        )

    return session.once(f"esp.{name}", build)


def check_esp_kernel(target: bakenn.TargetDescriptor, operation: str, session: Session) -> str:
    kernels = _kernels(_esp(session, target).artifacts)
    prefix = f"esp_nn.{target.target_id}.{operation}"
    selected = [kernel for kernel in kernels if kernel.startswith(prefix)]
    _expect(len(selected) == 1, f"ESP-NN {operation} was not selected for {target.target_id}: {kernels}")
    return f"target={target.target_id} kernel={selected[0]}"


def check_esp32_add_portable(session: Session) -> str:
    kernels = _kernels(_esp(session, bakenn.ESP32).artifacts)
    _expect("portable.add_s8.v1" in kernels, f"ESP32 Add did not keep the portable kernel: {kernels}")
    return "target=esp32 step=add kernel=portable.add_s8.v1"


def check_esp_sources(session: Session) -> str:
    artifacts = _esp(session, bakenn.ESP32_S3).artifacts
    sources = [path for path in artifacts.support_sources if path.is_file()]
    _expect(bool(sources) and len(sources) == len(artifacts.support_sources), "a bundled source is missing")
    return f"bundled_esp_nn_sources={len(sources)}"


def check_esp_license(session: Session) -> str:
    artifacts = _esp(session, bakenn.ESP32_S3).artifacts
    licenses = sorted(path.parent.name for path in artifacts.third_party_licenses if path.is_file())
    _expect(licenses == ["esp_nn"], f"bundled licenses: {licenses}")
    return f"bundled_licenses={','.join(licenses)}"


def check_dram_placement(session: Session) -> str:
    compiled = _esp(session, bakenn.ESP32_S3, requantization_in_dram=True)
    placed = compiled.memory_report.dram_constant_bytes
    _expect(bool(placed), "no requantization table was placed in DRAM")
    _expect("DRAM_ATTR" in compiled.artifacts.weights_source.read_text(), "the weights lack DRAM_ATTR")
    return f"dram_constant_bytes={placed} attribute=DRAM_ATTR"


def _idf(session: Session):  # type: ignore[no-untyped-def]
    return session.once(
        "idf",
        lambda: bakenn.export_esp_idf_project(
            _esp(session, bakenn.ESP32_S3).artifacts, bakenn.ESP32_S3, session.output / "esp/idf_project"
        ),
    )


def check_idf_file(relative: str, session: Session) -> str:
    path = _idf(session).root / relative
    _expect(path.is_file() and path.stat().st_size > 0, f"{relative} is missing or empty")
    return f"file={relative} present=1"


def check_idf_target(session: Session) -> str:
    description = json.loads((_idf(session).root / "bakenn_target.json").read_text())
    _expect(description["idf_target"] == "esp32s3", f"the project targets {description['idf_target']}")
    return f"file=bakenn_target.json idf_target={description['idf_target']}"


def check_idf_assembly(session: Session) -> str:
    component = (_idf(session).component / "CMakeLists.txt").read_text()
    _expect("esp_nn_add_s8_esp32s3.S" in component, "the ESP-NN Add assembly is not in the component")
    return "component_lists=esp_nn_add_s8_esp32s3.S"


def _exported_manifest(directory: Path) -> str:
    manifests = list(directory.glob("*_manifest.json"))
    _expect(len(manifests) == 1, f"{directory.name} holds {len(manifests)} manifests")
    files = bakenn.load_manifest(manifests[0])["artifact_inventory"]["files"]
    return f"hashed_files_verified={len(files)}"


def check_idf_manifest(session: Session) -> str:
    return _exported_manifest(_idf(session).component / "generated")


def check_idf_component(session: Session) -> str:
    component = Path(
        session.once(
            "idf_component",
            lambda: bakenn.export_esp_idf_component(
                _esp(session, bakenn.ESP32_S3).artifacts, bakenn.ESP32_S3, session.output / "esp/idf_component"
            ),
        )
    )
    _expect((component / "CMakeLists.txt").is_file(), "the component has no CMakeLists.txt")
    return "file=CMakeLists.txt present=1 " + _exported_manifest(component / "generated")


# --- Zephyr ------------------------------------------------------------------


def _zephyr(session: Session, board: str = "nrf52840dk_nrf52840"):  # type: ignore[no-untyped-def]
    return session.once(
        f"zephyr.{board}",
        lambda: bakenn.export_zephyr_project(
            _cortex_m4(session).artifacts, bakenn.CORTEX_M4, session.output / "zephyr" / board, board=board
        ),
    )


def check_zephyr_file(relative: str, session: Session) -> str:
    path = _zephyr(session).root / relative
    _expect(path.is_file() and path.stat().st_size > 0, f"{relative} is missing or empty")
    return f"file={relative} present=1"


def check_zephyr_manifest(session: Session) -> str:
    return _exported_manifest(_zephyr(session).generated)


def check_zephyr_board_option(session: Session) -> str:
    project = _zephyr(session, "nrf52dk_nrf52832")
    _expect(project.board == "nrf52dk_nrf52832", f"the project records board {project.board}")
    _expect((project.root / "CMakeLists.txt").is_file(), "the project has no CMakeLists.txt")
    return f"board={project.board} project_written=1"


# --- Arduino -----------------------------------------------------------------


def _arduino(session: Session):  # type: ignore[no-untyped-def]
    return session.once(
        "arduino",
        lambda: bakenn.export_arduino_library(session.compiled.artifacts, session.output / "arduino/library"),
    )


def _properties(root: Path) -> dict[str, str]:
    lines = (root / "library.properties").read_text().splitlines()
    return dict(line.split("=", 1) for line in lines if "=" in line)


def check_arduino_properties(session: Session) -> str:
    properties = _properties(_arduino(session).root)
    required = ("name", "version", "includes", "architectures")
    for key in required:
        _expect(bool(properties.get(key)), f"library.properties lacks {key}")
    return f"file=library.properties required_fields={len(required)}/{len(required)}"


def check_arduino_include(session: Session) -> str:
    library = _arduino(session)
    header = _properties(library.root)["includes"]
    _expect((library.source / header).is_file(), f"includes={header} names no file in src")
    return f"includes={header} header_present=1"


def check_arduino_any_board(session: Session) -> str:
    architectures = _properties(_arduino(session).root)["architectures"]
    _expect(architectures == "*", f"the library is limited to {architectures}")
    return f"architectures={architectures}"


def check_arduino_sources(session: Session) -> str:
    library = _arduino(session)
    sources = sorted(path.name for path in library.source.glob("*.c"))
    headers = sorted(path.name for path in library.source.glob("*.h"))
    _expect(len(sources) == 3 and len(headers) == 3, f"src holds {sources} and {headers}")
    return f"c_sources={len(sources)} headers={len(headers)}"


def check_arduino_example(session: Session) -> str:
    example = _arduino(session).example
    _expect(example.is_file() and example.suffix == ".ino", "the example sketch is missing")
    return f"example={example.parent.name}/{example.name} present=1"


def check_arduino_options(session: Session) -> str:
    library = session.once(
        "arduino.named",
        lambda: bakenn.export_arduino_library(
            session.compiled.artifacts, session.output / "arduino/named", name="MnistDigits", version="2.3.4"
        ),
    )
    properties = _properties(library.root)
    _expect(properties["name"] == "MnistDigits" and properties["version"] == "2.3.4", "options were not applied")
    return f"name={properties['name']} version={properties['version']}"


def _sketch(session: Session) -> list[int]:
    """Build the example sketch as C++ against a minimal Arduino.h and run it."""

    def build() -> list[int]:
        compiler = _require("c++")
        library = _arduino(session)
        directory = session.output / "arduino/build"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "Arduino.h").write_text(
            "#include <cstddef>\n#include <cstdint>\n#include <cstdio>\n"
            "struct SerialStub {\n"
            "  void begin(unsigned long) {}\n"
            "  void print(const char *text) { std::fputs(text, stdout); }\n"
            "  void print(char value) { std::fputc(value, stdout); }\n"
            "  void print(int value) { std::printf(\"%d\", value); }\n"
            "  void print(unsigned long value) { std::printf(\"%lu\", value); }\n"
            "  void println() { std::fputc('\\n', stdout); }\n"
            "};\n"
            "static SerialStub Serial;\n"
            "inline unsigned long micros() { return 0; }\n"
            "inline void delay(unsigned long) {}\n"
            "void setup();\nvoid loop();\n"
        )
        (directory / "main.cpp").write_text("void setup();\nvoid loop();\nint main() { setup(); loop(); }\n")
        objects = []
        for source in sorted(library.source.glob("*.c")):
            objects.append(directory / f"{source.stem}.o")
            subprocess.run(
                [
                    session.compiler, "-std=c11", "-O2", *WARNINGS, "-I", str(library.source),
                    "-c", str(source), "-o", str(objects[-1]),
                ],
                check=True,
            )
        # Arduino builds a sketch as C++ with Arduino.h included first.
        subprocess.run(
            [
                compiler, "-std=gnu++11", "-O2", "-Wall", "-Wextra", "-Werror",
                "-I", str(library.source), "-I", str(directory), "-include", "Arduino.h",
                "-x", "c++", "-c", str(library.example), "-o", str(directory / "sketch.o"),
            ],
            check=True,
        )
        subprocess.run(
            [
                compiler, str(directory / "main.cpp"), str(directory / "sketch.o"), *map(str, objects),
                "-o", str(directory / "sketch"),
            ],
            check=True,
        )
        printed = subprocess.run([str(directory / "sketch")], check=True, capture_output=True, text=True).stdout
        return [int(value) for value in printed.split("output=")[1].split()]

    return session.once("arduino.sketch", build)


def check_sketch_builds(session: Session) -> str:
    values = _sketch(session)
    _expect(len(values) == 10, f"the sketch printed {len(values)} values")
    return f"sketch_builds_as_cpp=1 printed_values={len(values)}"


def check_sketch_output(session: Session) -> str:
    plan = session.compiled.plan
    input_type = plan.tensors[plan.inputs[0]].tensor_type
    zero_input = np.full(input_type.shape, input_type.qparams.zero_point, dtype=np.int8)
    expected = bakenn.run_reference(plan, zero_input).reshape(-1).tolist()
    actual = _sketch(session)
    _expect(actual == expected, f"the sketch printed {actual}, the reference is {expected}")
    return f"sketch_output_matches_reference=1 values={len(actual)}"


# --- The numbered list -------------------------------------------------------

S3, LX6 = bakenn.ESP32_S3, bakenn.ESP32
_TESTS: tuple[tuple[str, str, Callable[[Session], str]], ...] = (
    ("install", "import the package and report its version", check_version),
    ("install", "report the Python, PyTorch and NumPy versions in use", check_runtime),
    ("install", "expose the documented entry points", check_entry_points),
    ("install", "state the same contract versions in the package and the C header", check_contract_versions),
    ("install", "provide the built-in target profiles", check_profiles),
    ("compile", "compile a PyTorch FP32 model into an execution plan", check_plan),
    ("compile", "turn the NCHW PyTorch input into an NHWC C input", check_layout),
    ("compile", "fold ReLU into the preceding integer step", check_relu_fused),
    ("compile", "emit weights as INT8 and biases as INT32 arrays", check_constant_types),
    ("compile", "declare every constant array const", check_constants_read_only),
    ("compile", "record the number of calibration samples", check_calibration_samples),
    ("compile", "record a valid FP32 range and INT8 scale for every calibrated tensor", check_calibration_ranges),
    ("compile", "produce identical files for identical inputs", check_deterministic),
    ("compile", "replace an earlier BakeNN output directory in place", check_recompile_in_place),
    ("compile", "turn the model name into a valid C identifier", check_symbol_name),
    *(
        ("operators", f"convert {label} and match the reference byte for byte", partial(check_operator, name, factory, shape, kernel))
        for label, name, factory, shape, kernel in OPERATORS
    ),
    *(
        ("networks", f"convert {label} and match the reference byte for byte", partial(check_network, name, factory, shape))
        for label, name, factory, shape in NETWORKS
    ),
    *(("files", f"write the {label}", partial(check_file, attribute)) for attribute, label in GENERATED_FILES),
    ("files", "write no file besides the ten generated ones", check_no_other_files),
    ("files", "declare the inference function in the public header", check_prototype),
    ("files", "publish the input and output byte counts as header macros", check_size_macros),
    ("files", "publish the input and output shapes as header macros", check_shape_macros),
    ("files", "publish the quantization parameters as header macros", check_qparam_macros),
    ("files", "accept the public header in a C++ translation unit", check_cpp_header),
    ("files", "list the three C sources in the CMake fragment", check_cmake_fragment),
    ("c-code", "call no heap function in the generated C", check_no_heap),
    ("c-code", "use no floating-point type in the generated C", check_no_float),
    ("c-code", "include only freestanding standard headers", check_includes),
    ("c-code", "compile as strict C99 without diagnostics", partial(check_strict_c, "c99")),
    ("c-code", "compile as strict C11 without diagnostics", partial(check_strict_c, "c11")),
    ("c-code", "give the expected outputs when built with -O0", partial(check_optimization, "-O0")),
    ("c-code", "give the expected outputs when built with -Os", partial(check_optimization, "-Os")),
    ("c-code", "give the expected outputs when built with -O3", partial(check_optimization, "-O3")),
    ("c-code", "run clean under AddressSanitizer and UndefinedBehaviorSanitizer", check_sanitizers),
    ("c-code", "write nothing outside the arena and the output buffer", check_guards),
    ("c-code", "leave the input buffer unchanged", check_input_unchanged),
    ("c-code", "give the same output whatever the arena held before", check_repeatable),
    ("outputs", "reproduce the frozen expected outputs with the generated C", check_frozen_outputs),
    ("outputs", "classify the frozen test images with the INT8 model", check_classification),
    ("outputs", "stay within one image of the FP32 model's accuracy", check_fp32_parity),
    ("outputs", "match the generated C with the Python integer reference", check_reference),
    ("outputs", "reproduce the frozen INT8 inputs with quantize_input", check_quantize_input),
    ("outputs", "apply the output scale and zero point in dequantize_output", check_dequantize_output),
    ("outputs", "expose every intermediate tensor with run_reference_trace", check_trace),
    ("accuracy", "report the number of samples compared", check_accuracy_samples),
    ("accuracy", "report one row per layer", check_accuracy_layers),
    ("accuracy", "report maximum, mean and RMS error of the output layer", check_accuracy_errors),
    ("accuracy", "count the values at the INT8 limits for every layer", check_accuracy_endpoints),
    ("accuracy", "limit the comparison with max_samples", check_accuracy_max_samples),
    ("memory", "report the arena size", check_arena_report),
    ("memory", "state the same arena size in the C header", check_arena_macro),
    ("memory", "state the same arena size in the manifest", check_arena_manifest),
    ("memory", "state the arena alignment in the report and the C header", check_arena_alignment),
    ("memory", "report the bytes of constant data", check_constant_bytes),
    ("memory", "report caller-owned input and output bytes apart from the arena", check_caller_io),
    ("memory", "report zero heap calls for the generated model", check_heap_report),
    ("memory", "place every activation buffer inside the arena", check_buffers_in_arena),
    ("memory", "reuse arena memory between layers", check_arena_reuse),
    ("memory", "report the peak working memory", check_peak),
    ("memory", "write the arena size into the text report file", check_memory_text),
    ("memory", "write the arena size into the JSON report file", check_memory_json),
    ("memory", "reject a model whose arena exceeds the SRAM budget", check_sram_rejected),
    ("memory", "write no file when the SRAM budget is exceeded", check_sram_nothing_written),
    ("memory", "accept a model whose arena equals the SRAM budget", check_sram_accepted),
    ("memory", "reject a model whose constants exceed the Flash budget", check_flash_rejected),
    ("memory", "write no file when the Flash budget is exceeded", check_flash_nothing_written),
    ("memory", "accept a model whose constants equal the Flash budget", check_flash_accepted),
    ("errors", "reject a model with an unsupported operator", check_unsupported_operator),
    ("errors", "attach an error code and suggestions to the unsupported operator", check_unsupported_diagnostic),
    ("errors", "write no file for a model with an unsupported operator", check_unsupported_nothing_written),
    ("errors", "reject a model left in training mode", check_training_mode),
    ("errors", "reject a model that is not a torch.nn.Module", check_not_a_module),
    ("errors", "reject an example input with batch size two", check_batch_size),
    ("errors", "reject an example input that is not float32", check_example_dtype),
    ("errors", "reject an example input that contains NaN", check_example_nan),
    ("errors", "reject two example inputs", check_two_inputs),
    ("errors", "reject an empty calibration set", check_empty_calibration),
    ("errors", "reject calibration data of another shape", check_calibration_shape),
    ("errors", "reject calibration data that contains NaN", check_calibration_nan),
    ("errors", "refuse an output path that is an existing file", check_output_is_file),
    ("errors", "refuse to replace a directory that holds foreign files", check_foreign_directory),
    ("errors", "reject an unknown target name and list the valid ones", check_unknown_target),
    ("errors", "reject REQUIRE_OPTIMIZED when a step has no optimized kernel", check_require_optimized),
    ("errors", "reject DRAM placement for a target without ESP-IDF", check_dram_needs_esp),
    ("errors", "reject a quantize_input array of the wrong shape", check_quantize_shape),
    ("errors", "reject a run_reference input that is not int8", check_reference_dtype),
    ("errors", "reject a run_reference input of the wrong shape", check_reference_shape),
    ("errors", "reject a dequantize_output array of the wrong shape", check_dequantize_shape),
    ("errors", "report a manifest file that does not exist", check_manifest_missing),
    ("errors", "report a manifest that is not valid JSON", check_manifest_malformed),
    ("errors", "refuse to build artifacts for another target", check_build_target_mismatch),
    ("errors", "refuse an ESP-IDF export for a target without ESP-IDF", check_idf_needs_esp),
    ("errors", "refuse a Zephyr export for a target other than Cortex-M4", check_zephyr_needs_cortex_m4),
    ("errors", "reject an unknown Zephyr board", check_zephyr_board),
    ("errors", "refuse an Arduino export of artifacts with vendor kernels", check_arduino_vendor_kernels),
    ("errors", "refuse to overwrite an existing Arduino library", check_arduino_existing),
    ("errors", "reject an Arduino library version that is not MAJOR.MINOR.PATCH", check_arduino_version),
    ("errors", "reject an invalid Arduino library name", check_arduino_name),
    ("integrity", "verify the generated files against the manifest", check_manifest_intact),
    ("integrity", "detect one flipped bit in a generated file", check_bit_flip),
    ("integrity", "detect a generated file that lost a byte", check_truncated_file),
    ("integrity", "detect a missing generated file", check_missing_file),
    ("integrity", "detect a file added to the generated directory", check_added_file),
    ("integrity", "detect an edited manifest", check_edited_manifest),
    ("integrity", "record the model, target and selected kernels in the manifest", check_manifest_identity),
    ("integrity", "record why each alternative kernel was not selected", check_manifest_rejections),
    ("integrity", "record SHA-256 fingerprints of the graph, plan and constants", check_fingerprints),
    ("integrity", "name the same arithmetic profile in the manifest and the C header", check_arithmetic_profile),
    ("integrity", "verify the hashes of the frozen evidence files", check_frozen_hashes),
    ("integrity", "rebuild the frozen generated C and reproduce its outputs", check_frozen_rebuild),
    ("portable-targets", "select only portable kernels by default", check_portable_kernels),
    *(("portable-targets", f"link the generated C for {cpu}", partial(check_core, cpu)) for cpu in ARM_CORES),
    ("portable-targets", "generate the same C sources for all seven Arm cores", check_identical_sources),
    ("portable-targets", "measure Flash and static SRAM of a linked image", check_image_size),
    *(
        ("portable-targets", f"generate portable C for the {name} profile", partial(check_profile, name))
        for name in ("cortex-m0plus", "rv32imc", "esp32c3")
    ),
    ("portable-targets", "link the generated C for RV32IMC", check_riscv_link),
    ("portable-targets", "keep portable kernels when CMSIS-NN is requested without DSP", check_vendor_fallback),
    ("cortex-m4", "select CMSIS-NN kernels for Cortex-M4", check_cmsis_selected),
    ("cortex-m4", "select the Cortex-M4 DSP kernel for Linear", check_dsp_linear),
    ("cortex-m4", "keep the portable kernel for a step without an optimized one", check_portable_step),
    ("cortex-m4", "bundle the CMSIS-NN sources the model needs", check_cmsis_sources),
    ("cortex-m4", "bundle the CMSIS-NN and CMSIS-Core licenses", check_cmsis_licenses),
    ("cortex-m4", "verify the bundled files against the manifest", check_cmsis_manifest),
    ("cortex-m4", "link for Cortex-M4 without undefined symbols", check_cortex_m4_undefined),
    ("cortex-m4", "link for Cortex-M4 without heap or floating-point symbols", check_cortex_m4_forbidden),
    ("cortex-m4", "keep portable kernels with the MEASURED policy and no measurements", check_policy_measured),
    ("esp32", "select the ESP-NN Conv2D kernel for ESP32-S3", partial(check_esp_kernel, S3, "conv2d_s8")),
    ("esp32", "select the ESP-NN depthwise Conv2D kernel for ESP32-S3", partial(check_esp_kernel, S3, "depthwise_conv2d_s8")),
    ("esp32", "select the ESP-NN Add kernel for ESP32-S3", partial(check_esp_kernel, S3, "add_s8")),
    ("esp32", "select the ESP-NN Conv2D kernel for ESP32", partial(check_esp_kernel, LX6, "conv2d_s8")),
    ("esp32", "select the ESP-NN depthwise Conv2D kernel for ESP32", partial(check_esp_kernel, LX6, "depthwise_conv2d_s8")),
    ("esp32", "keep the portable Add kernel for ESP32", check_esp32_add_portable),
    ("esp32", "bundle the ESP-NN sources the model needs", check_esp_sources),
    ("esp32", "bundle the ESP-NN license", check_esp_license),
    ("esp32", "place requantization tables in DRAM on request", check_dram_placement),
    ("esp32", "write the ESP-IDF project CMakeLists.txt", partial(check_idf_file, "CMakeLists.txt")),
    ("esp32", "write the ESP-IDF application main/main.c", partial(check_idf_file, "main/main.c")),
    ("esp32", "write the ESP-IDF model component CMakeLists.txt", partial(check_idf_file, "components/bakenn_model/CMakeLists.txt")),
    ("esp32", "write sdkconfig.defaults for the ESP-IDF project", partial(check_idf_file, "sdkconfig.defaults")),
    ("esp32", "record the ESP-IDF target in bakenn_target.json", check_idf_target),
    ("esp32", "list the ESP-NN Add assembly in the model component", check_idf_assembly),
    ("esp32", "verify the exported ESP-IDF model files against the manifest", check_idf_manifest),
    ("esp32", "export the model as a standalone ESP-IDF component", check_idf_component),
    ("zephyr", "write the Zephyr project CMakeLists.txt", partial(check_zephyr_file, "CMakeLists.txt")),
    ("zephyr", "write the Zephyr project prj.conf", partial(check_zephyr_file, "prj.conf")),
    ("zephyr", "write the Zephyr application src/main.c", partial(check_zephyr_file, "src/main.c")),
    ("zephyr", "verify the exported Zephyr model files against the manifest", check_zephyr_manifest),
    ("zephyr", "export the Zephyr project for another supported board", check_zephyr_board_option),
    ("arduino", "write library.properties with the required fields", check_arduino_properties),
    ("arduino", "name the model header in library.properties", check_arduino_include),
    ("arduino", "declare the Arduino library usable on every architecture", check_arduino_any_board),
    ("arduino", "copy the three C sources and three headers into src", check_arduino_sources),
    ("arduino", "write the example sketch Infer.ino", check_arduino_example),
    ("arduino", "apply the library name and version options", check_arduino_options),
    ("arduino", "build the example sketch as C++", check_sketch_builds),
    ("arduino", "print the reference output from the example sketch", check_sketch_output),
)
CHECKS = tuple(
    Check(f"T{number:03d}", group, title, run) for number, (group, title, run) in enumerate(_TESTS, start=1)
)


def _select(requested: list[str], parser: argparse.ArgumentParser) -> list[Check]:
    """Resolve test numbers and ``T010-T020`` ranges; keep the order of the list."""

    numbers = [check.identifier for check in CHECKS]
    chosen: set[str] = set()
    for item in requested:
        first, _, last = item.partition("-")
        if first not in numbers or (last and last not in numbers):
            parser.error(f"unknown test id: {item}")
        low, high = numbers.index(first), numbers.index(last or first)
        if low > high:
            parser.error(f"empty test range: {item}")
        chosen.update(numbers[low : high + 1])
    return [check for check in CHECKS if not requested or check.identifier in chosen]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checks", nargs="*", help="test ids or ranges such as T010-T020; default is all")
    parser.add_argument("--list", action="store_true", help="print ids, groups and titles, then exit")
    parser.add_argument("--strict", action="store_true", help="treat a skipped test as a failure")
    parser.add_argument("--cc", default="cc", help="host C compiler")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "build/functional_checks",
        help="scratch directory; an earlier one from this script is replaced",
    )
    arguments = parser.parse_args()
    if arguments.list:
        for check in CHECKS:
            print(f"{check.identifier}  [{check.group}] {check.title}")
        return 0
    selected = _select(arguments.checks, parser)

    # Only a directory this script created is ever deleted.
    output = arguments.output.resolve()
    marker = output / ".bakenn_functional_checks"
    if output.exists() and any(output.iterdir()) and not marker.is_file():
        parser.error(f"--output is not an earlier functional-checks directory: {output}")
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    marker.write_text("scratch directory of scripts/functional_checks.py\n")
    session = Session(output, arguments.cc)

    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    group = None
    for check in selected:
        if check.group != group:
            group = check.group
            print(f"== {group} ==", flush=True)
        try:
            status, detail = "PASS", run_check(check, session)
        except Skip as skipped:
            status, detail = "SKIP", str(skipped)
        except Exception as error:  # noqa: BLE001 - every failure becomes one result line
            status, detail = "FAIL", f"{type(error).__name__}: {error}"
        counts[status] += 1
        print(f"{check.identifier} {status}  {check.title}\n           {detail}", flush=True)
    print(f"\nRESULT pass={counts['PASS']} fail={counts['FAIL']} skip={counts['SKIP']} of {len(selected)}")
    return 1 if counts["FAIL"] or (arguments.strict and counts["SKIP"]) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted before all selected tests ran", file=sys.stderr)
        raise SystemExit(130) from None
