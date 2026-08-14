#!/usr/bin/env python3
"""Train a compact MNIST CNN and run the complete BakeNN PTQ-to-C path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

import bakenn  # noqa: E402


class MNISTNet(nn.Module):
    """Small enough for a portable-C smoke test, large enough for useful accuracy."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Linear(8 * 7 * 7, 10)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.pool(torch.relu(self.conv1(value)))
        value = self.pool(torch.relu(self.conv2(value)))
        return self.classifier(torch.flatten(value, 1))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/private/tmp/bakenn-mnist-data"))
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY / "examples/mnist/build")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--calibration-per-class", type=int, default=16)
    parser.add_argument("--reference-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    return parser.parse_args()


def accuracy(model: nn.Module, loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]) -> float:
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum())
            total += int(labels.numel())
    return correct / total


def train(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    epochs: int,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    loss_function = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        loss_sum = 0.0
        sample_count = 0
        started = time.perf_counter()
        for images, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * int(labels.numel())
            sample_count += int(labels.numel())
        elapsed = time.perf_counter() - started
        print(
            f"epoch {epoch + 1}/{epochs}: loss={loss_sum / sample_count:.5f}, "
            f"seconds={elapsed:.2f}",
            flush=True,
        )


def calibration_samples(
    dataset: datasets.MNIST,
    per_class: int,
) -> torch.Tensor:
    selected: list[torch.Tensor] = []
    counts = [0] * 10
    for image, label in dataset:
        digit = int(label)
        if counts[digit] < per_class:
            selected.append(image)
            counts[digit] += 1
        if all(count == per_class for count in counts):
            break
    if not selected or not all(count == per_class for count in counts):
        raise RuntimeError("could not construct class-balanced calibration data")
    return torch.stack(selected)


def quantize_mnist_corpus(plan: object, pixels: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of bakenn.quantize_input for uint8 MNIST pixels."""

    input_name = plan.inputs[0]
    tensor_type = plan.tensors[input_name].tensor_type
    qparams = tensor_type.qparams
    values = np.asarray(pixels, dtype=np.float32)[..., np.newaxis] / np.float32(255.0)
    centered = values / np.float32(qparams.scale)
    rounded = np.where(
        centered >= 0.0,
        np.floor(centered + np.float32(0.5)),
        np.ceil(centered - np.float32(0.5)),
    )
    rounded += qparams.zero_point
    result = np.clip(rounded, -128, 127).astype(np.int8)
    expected = (pixels.shape[0], *tensor_type.shape[1:])
    if result.shape != expected:
        raise RuntimeError(f"quantized corpus has shape {result.shape}, expected {expected}")
    for index in range(min(16, result.shape[0])):
        public_result = bakenn.quantize_input(plan, values[index : index + 1])
        np.testing.assert_array_equal(result[index : index + 1], public_result)
    return result


def build_runner(compiled: object, compiler: str) -> Path:
    if shutil.which(compiler) is None:
        raise RuntimeError(f"C compiler not found: {compiler}")
    artifacts = compiled.artifacts
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    symbol = manifest["model"]
    macro = symbol.upper()
    runner_source = artifacts.output_dir / "mnist_runner.c"
    runner_source.write_text(
        f'''#include "{artifacts.header.name}"
#include <stddef.h>
#include <stdio.h>

#define ARENA_STORAGE_SIZE ({macro}_ARENA_SIZE == 0u ? 1u : {macro}_ARENA_SIZE)

int main(void) {{
    _Alignas({macro}_ARENA_ALIGNMENT) static uint8_t arena_storage[ARENA_STORAGE_SIZE];
    static int8_t input[{macro}_INPUT_SIZE];
    static int8_t output[{macro}_OUTPUT_SIZE];
    uint8_t *arena = {macro}_ARENA_SIZE == 0u ? NULL : arena_storage;
    while (fread(input, 1u, {macro}_INPUT_BYTES, stdin) == {macro}_INPUT_BYTES) {{
        {symbol}_infer(arena, input, output);
        if (fwrite(output, 1u, {macro}_OUTPUT_BYTES, stdout) != {macro}_OUTPUT_BYTES) {{
            return 2;
        }}
    }}
    return ferror(stdin) ? 3 : 0;
}}
''',
        encoding="utf-8",
    )
    executable = artifacts.output_dir / "mnist_runner"
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-O3",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            str(artifacts.model_source),
            str(artifacts.weights_source),
            str(artifacts.kernels_source),
            str(runner_source),
            "-I",
            str(artifacts.output_dir),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
    )
    return executable


def main() -> int:
    arguments = parse_arguments()
    if arguments.epochs <= 0 or arguments.calibration_per_class <= 0:
        raise ValueError("epochs and calibration-per-class must be positive")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    transform = transforms.ToTensor()
    print(f"loading MNIST from {arguments.data_dir}", flush=True)
    training_set = datasets.MNIST(arguments.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(arguments.data_dir, train=False, download=True, transform=transform)
    generator = torch.Generator().manual_seed(arguments.seed)
    training_loader = DataLoader(
        training_set,
        batch_size=128,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)

    model = MNISTNet()
    train(model, training_loader, arguments.epochs)
    model.eval()
    torch.save(model.state_dict(), arguments.output_dir / "mnist_fp32.pt")
    fp32_accuracy = accuracy(model, test_loader)
    print(f"FP32 test accuracy: {fp32_accuracy * 100.0:.2f}%", flush=True)

    calibration = calibration_samples(training_set, arguments.calibration_per_class)
    print(f"calibrating BakeNN with {len(calibration)} balanced samples", flush=True)
    compile_started = time.perf_counter()
    compiled = bakenn.compile_torch_ptq(
        model,
        calibration[:1],
        calibration,
        arguments.output_dir / "generated",
        name="mnist",
    )
    compile_seconds = time.perf_counter() - compile_started
    print(f"BakeNN PTQ and C generation: {compile_seconds:.2f}s", flush=True)

    test_pixels = test_set.data.numpy()
    labels = test_set.targets.numpy()
    input_codes = quantize_mnist_corpus(compiled.plan, test_pixels)
    runner = build_runner(compiled, arguments.cc)
    c_started = time.perf_counter()
    completed = subprocess.run(
        [str(runner)],
        input=input_codes.tobytes(),
        capture_output=True,
        check=True,
    )
    c_seconds = time.perf_counter() - c_started
    output_shape = compiled.plan.tensors[compiled.plan.outputs[0]].tensor_type.shape
    output_size = int(np.prod(output_shape))
    expected_bytes = len(test_set) * output_size
    if len(completed.stdout) != expected_bytes:
        raise RuntimeError(
            f"C runner returned {len(completed.stdout)} bytes, expected {expected_bytes}"
        )
    c_outputs = np.frombuffer(completed.stdout, dtype=np.int8).reshape(len(test_set), output_size)
    c_accuracy = float(np.mean(np.argmax(c_outputs, axis=1) == labels))
    print(f"generated C test accuracy: {c_accuracy * 100.0:.2f}%", flush=True)

    reference_count = min(arguments.reference_samples, len(test_set))
    reference_outputs = np.concatenate(
        [
            bakenn.run_reference(compiled.plan, input_codes[index : index + 1])
            for index in range(reference_count)
        ],
        axis=0,
    ).reshape(reference_count, output_size)
    mismatched_bytes = int(np.count_nonzero(reference_outputs != c_outputs[:reference_count]))
    if mismatched_bytes:
        raise RuntimeError(f"Python INT8 and generated C differ in {mismatched_bytes} bytes")
    print(
        f"Python INT8 vs C: {reference_count * output_size} bytes compared, 0 mismatches",
        flush=True,
    )

    manifest = json.loads(compiled.artifacts.manifest.read_text(encoding="utf-8"))
    report = {
        "seed": arguments.seed,
        "epochs": arguments.epochs,
        "training_samples": len(training_set),
        "test_samples": len(test_set),
        "calibration_samples": len(calibration),
        "reference_samples": reference_count,
        "torch_version": torch.__version__,
        "bakenn_version": bakenn.__version__,
        "compiler": arguments.cc,
        "fp32_accuracy": fp32_accuracy,
        "generated_c_accuracy": c_accuracy,
        "accuracy_drop_percentage_points": (fp32_accuracy - c_accuracy) * 100.0,
        "python_c_compared_bytes": reference_count * output_size,
        "python_c_mismatched_bytes": mismatched_bytes,
        "ptq_compile_seconds_host": compile_seconds,
        "c_full_test_seconds_host_including_pipes": c_seconds,
        "arena_bytes": manifest["arena_bytes"],
        "constant_bytes": manifest["constant_bytes"],
        "input": manifest["input"],
        "output": manifest["output"],
        "arithmetic_profile": manifest["arithmetic_profile"],
        "generated_model_symbol": manifest["model"],
    }
    report_path = arguments.output_dir / "mnist_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
