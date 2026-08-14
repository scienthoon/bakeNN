#!/usr/bin/env python3
"""Train six small real-data classifiers and exercise BakeNN end to end."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

import bakenn  # noqa: E402


class MNISTMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(28 * 28, 64)
        self.output = nn.Linear(64, 10)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.flatten(value, 1)
        return self.output(torch.relu(self.hidden(value)))


class MNISTCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.output = nn.Linear(8 * 7 * 7, 10)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.pool(torch.relu(self.conv1(value)))
        value = self.pool(torch.relu(self.conv2(value)))
        return self.output(torch.flatten(value, 1))


class CIFARBottleneck(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 12, 3, padding=1)
        self.reduce = nn.Conv2d(12, 6, 1)
        self.spatial = nn.Conv2d(6, 6, 3, padding=1)
        self.expand = nn.Conv2d(6, 12, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output = nn.Linear(12, 10)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = torch.relu(self.stem(value))
        value = torch.relu(self.reduce(residual))
        value = torch.relu(self.spatial(value))
        value = torch.relu(self.expand(value) + residual)
        return self.output(torch.flatten(self.pool(value), 1))


class CIFARDenseConcat(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1)
        self.grow1 = nn.Conv2d(8, 4, 3, padding=1)
        self.grow2 = nn.Conv2d(12, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output = nn.Linear(16, 10)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        features = [torch.relu(self.stem(value))]
        first_input = torch.cat(features, dim=1)
        features.append(torch.relu(self.grow1(first_input)))
        features.append(torch.relu(self.grow2(torch.cat(features, dim=1))))
        value = torch.cat(features, dim=1)
        return self.output(torch.flatten(self.pool(value), 1))


class CIFARInception(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 1)
        self.branch1 = nn.Conv2d(8, 4, 1)
        self.branch3 = nn.Conv2d(8, 4, 3, padding=1)
        self.pool_project = nn.Conv2d(8, 4, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output = nn.Linear(12, 10)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.relu(self.stem(value))
        branch1 = torch.relu(self.branch1(value))
        branch3 = torch.relu(self.branch3(value))
        pooled = torch.nn.functional.avg_pool2d(
            value,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )
        pooled = torch.relu(self.pool_project(pooled))
        value = torch.cat((branch1, branch3, pooled), dim=1)
        return self.output(torch.flatten(self.pool(value), 1))


class CIFARFire(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1)
        self.squeeze = nn.Conv2d(8, 4, 1)
        self.expand1 = nn.Conv2d(4, 4, 1)
        self.expand3 = nn.Conv2d(4, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output = nn.Linear(8, 10)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.relu(self.stem(value))
        squeezed = torch.relu(self.squeeze(value))
        value = torch.cat(
            (
                torch.relu(self.expand1(squeezed)),
                torch.relu(self.expand3(squeezed)),
            ),
            dim=1,
        )
        return self.output(torch.flatten(self.pool(value), 1))


@dataclass(frozen=True)
class Experiment:
    name: str
    dataset: str
    model_factory: Callable[[], nn.Module]


EXPERIMENTS = (
    Experiment("mnist_mlp", "mnist", MNISTMLP),
    Experiment("mnist_cnn", "mnist", MNISTCNN),
    Experiment("cifar10_bottleneck", "cifar10", CIFARBottleneck),
    Experiment("cifar10_dense_concat", "cifar10", CIFARDenseConcat),
    Experiment("cifar10_inception", "cifar10", CIFARInception),
    Experiment("cifar10_fire", "cifar10", CIFARFire),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/private/tmp/bakenn-datasets"))
    parser.add_argument(
        "--cifar-dir",
        type=Path,
        default=Path("/private/tmp/bakenn-datasets/cifar10-fastai"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY / "examples/training_matrix/build",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--calibration-per-class", type=int, default=10)
    parser.add_argument("--reference-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument(
        "--models",
        nargs="*",
        choices=[experiment.name for experiment in EXPERIMENTS],
        default=[experiment.name for experiment in EXPERIMENTS],
    )
    return parser.parse_args()


def datasets_for(
    name: str,
    data_dir: Path,
    cifar_dir: Path,
) -> tuple[Dataset[tuple[torch.Tensor, int]], Dataset[tuple[torch.Tensor, int]]]:
    transform = transforms.ToTensor()
    if name == "mnist":
        return (
            datasets.MNIST(data_dir, train=True, download=True, transform=transform),
            datasets.MNIST(data_dir, train=False, download=True, transform=transform),
        )
    train_dir, test_dir = cifar_dir / "train", cifar_dir / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise RuntimeError(
            f"CIFAR-10 ImageFolder data is missing under {cifar_dir}; see README.md"
        )
    return datasets.ImageFolder(train_dir, transform), datasets.ImageFolder(test_dir, transform)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    epochs: int,
) -> tuple[float, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    loss_function = nn.CrossEntropyLoss()
    final_loss = 0.0
    total_seconds = 0.0
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        started = time.perf_counter()
        for images, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()
            batch = int(labels.numel())
            loss_sum += float(loss.detach()) * batch
            sample_count += batch
        elapsed = time.perf_counter() - started
        total_seconds += elapsed
        final_loss = loss_sum / sample_count
        print(
            f"  epoch {epoch + 1}/{epochs}: loss={final_loss:.5f}, seconds={elapsed:.2f}",
            flush=True,
        )
    return final_loss, total_seconds


def accuracy(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            prediction = model(images).argmax(dim=1)
            correct += int((prediction == labels).sum())
            total += int(labels.numel())
    return correct / total


def balanced_calibration(
    dataset: Dataset[tuple[torch.Tensor, int]],
    per_class: int,
) -> torch.Tensor:
    selected: list[torch.Tensor] = []
    counts = [0] * 10
    for image, label in dataset:
        class_index = int(label)
        if counts[class_index] < per_class:
            selected.append(image)
            counts[class_index] += 1
        if all(count == per_class for count in counts):
            break
    if not all(count == per_class for count in counts):
        raise RuntimeError(f"could not select balanced calibration data: {counts}")
    return torch.stack(selected)


def _quantize_batch(plan: object, images: torch.Tensor) -> np.ndarray:
    tensor_type = plan.tensors[plan.inputs[0]].tensor_type
    qparams = tensor_type.qparams
    source = images.detach().numpy()
    if source.ndim == 4:
        source = np.ascontiguousarray(source.transpose(0, 2, 3, 1))
    centered = source.astype(np.float32) / np.float32(qparams.scale)
    rounded = np.where(
        centered >= 0.0,
        np.floor(centered + np.float32(0.5)),
        np.ceil(centered - np.float32(0.5)),
    )
    result = np.clip(rounded + qparams.zero_point, -128, 127).astype(np.int8)
    if result.shape[1:] != tensor_type.shape[1:]:
        raise RuntimeError(
            f"canonical input shape {result.shape[1:]} != ABI {tensor_type.shape[1:]}"
        )
    return result


def quantized_test_corpus(
    plan: object,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[np.ndarray, np.ndarray]:
    codes: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, batch_labels in loader:
        codes.append(_quantize_batch(plan, images))
        labels.append(batch_labels.numpy())
    result = np.concatenate(codes, axis=0)
    all_labels = np.concatenate(labels, axis=0)
    return result, all_labels


def build_runner(compiled: object, compiler: str) -> Path:
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        raise RuntimeError(f"C compiler not found: {compiler}")
    artifacts = compiled.artifacts
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    symbol = manifest["model"]
    macro = symbol.upper()
    source = artifacts.output_dir / "dataset_runner.c"
    source.write_text(
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
    executable = artifacts.output_dir / "dataset_runner"
    subprocess.run(
        [
            compiler_path,
            "-std=c11",
            "-O3",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            str(artifacts.model_source),
            str(artifacts.weights_source),
            str(artifacts.kernels_source),
            str(source),
            "-I",
            str(artifacts.output_dir),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
    )
    return executable


def run_experiment(
    experiment: Experiment,
    arguments: argparse.Namespace,
) -> dict[str, object]:
    print(f"\n[{experiment.name}] loading {experiment.dataset}", flush=True)
    training_set, test_set = datasets_for(
        experiment.dataset,
        arguments.data_dir,
        arguments.cifar_dir,
    )
    generator = torch.Generator().manual_seed(arguments.seed)
    training_loader = DataLoader(
        training_set,
        batch_size=arguments.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=512,
        shuffle=False,
        num_workers=0,
    )
    torch.manual_seed(arguments.seed)
    model = experiment.model_factory()
    loss, training_seconds = train_one_epoch(model, training_loader, arguments.epochs)
    fp32_accuracy = accuracy(model, test_loader)
    print(f"  FP32 accuracy: {fp32_accuracy * 100.0:.2f}%", flush=True)

    model.eval()
    output_dir = arguments.output_dir / experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "fp32_state.pt")
    calibration = balanced_calibration(training_set, arguments.calibration_per_class)
    compile_started = time.perf_counter()
    compiled = bakenn.compile_torch_ptq(
        model,
        calibration[:1],
        calibration,
        output_dir / "generated",
        name=experiment.name,
    )
    compile_seconds = time.perf_counter() - compile_started
    print(f"  PTQ + C generation: {compile_seconds:.2f}s", flush=True)

    input_codes, labels = quantized_test_corpus(compiled.plan, test_loader)
    first_images, _ = next(iter(test_loader))
    first_real = first_images[:1].numpy().transpose(0, 2, 3, 1)
    np.testing.assert_array_equal(
        input_codes[:1],
        bakenn.quantize_input(compiled.plan, np.ascontiguousarray(first_real)),
    )
    runner = build_runner(compiled, arguments.cc)
    c_started = time.perf_counter()
    completed = subprocess.run(
        [str(runner)],
        input=input_codes.tobytes(),
        capture_output=True,
        check=True,
    )
    c_seconds = time.perf_counter() - c_started
    output_type = compiled.plan.tensors[compiled.plan.outputs[0]].tensor_type
    output_size = output_type.numel
    expected_bytes = len(test_set) * output_size
    if len(completed.stdout) != expected_bytes:
        raise RuntimeError(
            f"generated C returned {len(completed.stdout)} bytes, expected {expected_bytes}"
        )
    c_outputs = np.frombuffer(completed.stdout, dtype=np.int8).reshape(len(test_set), output_size)
    int8_accuracy = float(np.mean(np.argmax(c_outputs, axis=1) == labels))
    reference_count = min(arguments.reference_samples, len(test_set))
    reference = np.concatenate(
        [
            bakenn.run_reference(compiled.plan, input_codes[index : index + 1])
            for index in range(reference_count)
        ],
        axis=0,
    ).reshape(reference_count, output_size)
    mismatches = int(np.count_nonzero(reference != c_outputs[:reference_count]))
    if mismatches:
        raise RuntimeError(f"Python INT8 and generated C differ in {mismatches} bytes")
    print(
        f"  generated-C INT8 accuracy: {int8_accuracy * 100.0:.2f}% "
        f"(drop={(fp32_accuracy - int8_accuracy) * 100.0:+.2f} pp)",
        flush=True,
    )
    print(
        f"  Python INT8 vs C: {reference_count * output_size} bytes, 0 mismatches",
        flush=True,
    )
    manifest = json.loads(compiled.artifacts.manifest.read_text(encoding="utf-8"))
    report: dict[str, object] = {
        "model": experiment.name,
        "dataset": experiment.dataset,
        "seed": arguments.seed,
        "epochs": arguments.epochs,
        "training_samples": len(training_set),
        "test_samples": len(test_set),
        "calibration_samples": len(calibration),
        "final_training_loss": loss,
        "training_seconds_cpu": training_seconds,
        "fp32_accuracy": fp32_accuracy,
        "generated_c_int8_accuracy": int8_accuracy,
        "accuracy_drop_percentage_points": (fp32_accuracy - int8_accuracy) * 100.0,
        "python_c_compared_bytes": reference_count * output_size,
        "python_c_mismatched_bytes": mismatches,
        "ptq_compile_seconds_host": compile_seconds,
        "c_full_test_seconds_host_including_pipes": c_seconds,
        "arena_bytes": manifest["arena_bytes"],
        "constant_bytes": manifest["constant_bytes"],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def write_summary(reports: list[dict[str, object]], output_dir: Path) -> None:
    payload = {
        "bakenn_version": bakenn.__version__,
        "torch_version": torch.__version__,
        "device": "cpu",
        "reports": reports,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# One-epoch real-data matrix",
        "",
        (
            f"All entries use the complete training split for {int(reports[0]['epochs'])} epoch(s) "
            f"and {int(reports[0]['calibration_samples'])} class-balanced PTQ images."
        ),
        "Generated C runs all 10,000 test images; host timing is not an MCU benchmark.",
        "",
        "| Model | Dataset | FP32 | INT8 C | FP32-INT8 (pp) | Train s | Arena | Constants |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        lines.append(
            "| {model} | {dataset} | {fp32:.2f}% | {int8:.2f}% | {drop:+.2f} | "
            "{seconds:.1f} | {arena} B | {constants} B |".format(
                model=report["model"],
                dataset=report["dataset"],
                fp32=float(report["fp32_accuracy"]) * 100.0,
                int8=float(report["generated_c_int8_accuracy"]) * 100.0,
                drop=float(report["accuracy_drop_percentage_points"]),
                seconds=float(report["training_seconds_cpu"]),
                arena=int(report["arena_bytes"]),
                constants=int(report["constant_bytes"]),
            )
        )
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    arguments = parse_arguments()
    if arguments.epochs <= 0 or arguments.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if arguments.calibration_per_class <= 0 or arguments.reference_samples <= 0:
        raise ValueError("calibration-per-class and reference-samples must be positive")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    selected = [item for item in EXPERIMENTS if item.name in arguments.models]
    reports: list[dict[str, object]] = []
    for experiment in selected:
        reports.append(run_experiment(experiment, arguments))
        write_summary(reports, arguments.output_dir)
    print(f"\nsummary: {arguments.output_dir / 'RESULTS.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
