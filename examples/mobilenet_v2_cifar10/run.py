#!/usr/bin/env python3
"""Train MobileNetV2-0.25 on CIFAR-10 and deploy it with BakeNN."""

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
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

import bakenn  # noqa: E402
from bakenn.targets import ESP32_S3  # noqa: E402


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def mobilenet_v2_quarter() -> nn.Module:
    """Return the exact fixed-shape representative model used by this example."""

    return models.mobilenet_v2(
        weights=None,
        width_mult=0.25,
        num_classes=10,
        dropout=0.1,
    )


def _datasets(data_dir: Path) -> tuple[Dataset, Dataset, Dataset]:
    train_transform = transforms.Compose(
        (
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        )
    )
    evaluation_transform = transforms.Compose(
        (
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        )
    )
    train_root = data_dir / "train"
    test_root = data_dir / "test"
    if not train_root.is_dir() or not test_root.is_dir():
        raise RuntimeError(
            f"CIFAR-10 ImageFolder data is missing under {data_dir}; see README.md"
        )
    return (
        datasets.ImageFolder(train_root, train_transform),
        datasets.ImageFolder(train_root, evaluation_transform),
        datasets.ImageFolder(test_root, evaluation_transform),
    )


def _limited(dataset: Dataset, maximum: int, seed: int) -> Dataset:
    if maximum <= 0 or maximum >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:maximum].tolist()
    return Subset(dataset, indices)


def _device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _train(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
) -> tuple[float, float]:
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    loss_function = nn.CrossEntropyLoss()
    final_loss = 0.0
    elapsed_total = 0.0
    for epoch in range(epochs):
        started = time.perf_counter()
        loss_sum = 0.0
        sample_count = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()
            batch = int(labels.numel())
            loss_sum += float(loss.detach()) * batch
            sample_count += batch
        elapsed = time.perf_counter() - started
        elapsed_total += elapsed
        final_loss = loss_sum / sample_count
        print(
            f"epoch {epoch + 1}/{epochs}: loss={final_loss:.5f}, seconds={elapsed:.1f}",
            flush=True,
        )
    model.cpu().eval()
    return final_loss, elapsed_total


def _accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum())
            total += int(labels.numel())
    return correct / total


def _balanced_calibration(dataset: Dataset, per_class: int) -> torch.Tensor:
    selected: list[torch.Tensor] = []
    counts = [0] * 10
    for image, label in dataset:
        index = int(label)
        if counts[index] < per_class:
            selected.append(image)
            counts[index] += 1
        if all(count == per_class for count in counts):
            break
    if not all(count == per_class for count in counts):
        raise RuntimeError(f"could not construct balanced calibration set: {counts}")
    return torch.stack(selected)


def _quantize_images(plan: object, images: torch.Tensor) -> np.ndarray:
    tensor_type = plan.tensors[plan.inputs[0]].tensor_type
    qparams = tensor_type.qparams
    source = np.ascontiguousarray(images.numpy().transpose(0, 2, 3, 1))
    centered = source.astype(np.float32) / np.float32(qparams.scale)
    rounded = np.where(
        centered >= 0.0,
        np.floor(centered + np.float32(0.5)),
        np.ceil(centered - np.float32(0.5)),
    )
    return np.clip(rounded + qparams.zero_point, -128, 127).astype(np.int8)


def _test_corpus(plan: object, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    inputs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, batch_labels in loader:
        inputs.append(_quantize_images(plan, images))
        labels.append(batch_labels.numpy())
    return np.concatenate(inputs), np.concatenate(labels)


def _build_host_runner(compiled: object, compiler: str) -> Path:
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        raise RuntimeError(f"C compiler not found: {compiler}")
    artifacts = compiled.artifacts
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    symbol = str(manifest["model"])
    macro = symbol.upper()
    source = artifacts.output_dir / "dataset_runner.c"
    source.write_text(
        f'''#include "{artifacts.header.name}"
#include <stdint.h>
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
    command = [
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
        *(str(item) for item in artifacts.support_sources),
        str(source),
    ]
    for include_dir in (artifacts.output_dir, *artifacts.support_include_dirs):
        command.extend(("-I", str(include_dir)))
    command.extend(("-o", str(executable)))
    subprocess.run(command, check=True, capture_output=True)
    return executable


def _evaluate_generated_c(
    compiled: object,
    loader: DataLoader,
    compiler: str,
    reference_samples: int,
) -> tuple[float, int, int, float]:
    input_codes, labels = _test_corpus(compiled.plan, loader)
    runner = _build_host_runner(compiled, compiler)
    started = time.perf_counter()
    process = subprocess.run(
        [str(runner)],
        input=input_codes.tobytes(),
        capture_output=True,
        check=True,
    )
    elapsed = time.perf_counter() - started
    output_size = compiled.plan.tensors[compiled.plan.outputs[0]].tensor_type.numel
    expected_bytes = len(labels) * output_size
    if len(process.stdout) != expected_bytes:
        raise RuntimeError(
            f"generated C produced {len(process.stdout)} bytes, expected {expected_bytes}"
        )
    outputs = np.frombuffer(process.stdout, dtype=np.int8).reshape(len(labels), output_size)
    int8_accuracy = float(np.mean(np.argmax(outputs, axis=1) == labels))
    compared = min(reference_samples, len(labels))
    reference = np.concatenate(
        [
            bakenn.run_reference(
                compiled.plan,
                input_codes[index : index + 1],
            )
            for index in range(compared)
        ],
        axis=0,
    ).reshape(compared, output_size)
    mismatches = int(np.count_nonzero(reference != outputs[:compared]))
    if mismatches:
        raise RuntimeError(f"Python INT8 and generated C differ in {mismatches} bytes")
    return int8_accuracy, compared * output_size, mismatches, elapsed


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/private/tmp/bakenn-datasets/cifar10-fastai"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "build/mobilenet_v2_cifar10",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--calibration-per-class", type=int, default=2)
    parser.add_argument("--reference-samples", type=int, default=16)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="PyTorch CPU threads; one is fastest on the measured macOS host",
    )
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    if arguments.epochs <= 0:
        raise ValueError("epochs must be positive")
    if (
        arguments.batch_size <= 0
        or arguments.calibration_per_class <= 0
        or arguments.threads <= 0
    ):
        raise ValueError("batch-size, calibration-per-class and threads must be positive")
    if arguments.reference_samples <= 0:
        raise ValueError("reference-samples must be positive")
    arguments.output.mkdir(parents=True, exist_ok=True)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.set_num_threads(arguments.threads)

    train_data, calibration_data, test_data = _datasets(arguments.data_dir)
    train_data = _limited(train_data, arguments.max_train_samples, arguments.seed)
    test_data = _limited(test_data, arguments.max_test_samples, arguments.seed + 1)
    generator = torch.Generator().manual_seed(arguments.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=arguments.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)
    device = _device(arguments.device)
    print(f"training MobileNetV2-0.25 on {device} with {len(train_data)} images", flush=True)

    model = mobilenet_v2_quarter()
    loss, training_seconds = _train(model, train_loader, device, arguments.epochs)
    fp32_accuracy = _accuracy(model, test_loader)
    checkpoint = arguments.output / "mobilenet_v2_025_cifar10_fp32.pt"
    torch.save(model.state_dict(), checkpoint)
    print(f"FP32 test accuracy: {fp32_accuracy * 100.0:.2f}%", flush=True)

    calibration = _balanced_calibration(calibration_data, arguments.calibration_per_class)
    compile_started = time.perf_counter()
    portable = bakenn.compile_torch_ptq(
        model,
        calibration[:1],
        calibration,
        arguments.output / "generated_portable",
        name="mobilenet_v2_025_cifar10",
        backend_options=bakenn.CBackendOptions(
            kernel_policy=bakenn.KernelPolicy.AUTO,
        ),
    )
    ptq_seconds = time.perf_counter() - compile_started
    print(f"PTQ + portable C generation: {ptq_seconds:.1f}s", flush=True)

    int8_accuracy, compared_bytes, mismatches, c_seconds = _evaluate_generated_c(
        portable,
        test_loader,
        arguments.cc,
        arguments.reference_samples,
    )
    print(
        f"generated-C INT8 accuracy: {int8_accuracy * 100.0:.2f}% "
        f"(FP32-INT8={(fp32_accuracy - int8_accuracy) * 100.0:+.2f} pp)",
        flush=True,
    )
    print(f"Python INT8 vs C: {compared_bytes} bytes, {mismatches} mismatches", flush=True)

    esp32s3 = bakenn.compile(
        portable.graph,
        arguments.output / "generated_esp32s3",
        model_name="mobilenet_v2_025_cifar10",
        target=ESP32_S3,
        backend_options=bakenn.CBackendOptions(
            kernel_policy=bakenn.KernelPolicy.AUTO,
            enable_esp_nn=True,
            target=ESP32_S3,
        ),
    )
    project = bakenn.export_esp_idf_project(
        esp32s3.artifacts,
        ESP32_S3,
        arguments.output / "esp_idf",
    )
    optimized = sum(
        int(selection.optimized)
        for selection in esp32s3.artifacts.backend_plan.selections
    )
    report = {
        "schema_version": 1,
        "model": "torchvision MobileNetV2 width_mult=0.25 num_classes=10",
        "dataset": "CIFAR-10 32x32 RGB",
        "seed": arguments.seed,
        "epochs": arguments.epochs,
        "training_samples": len(train_data),
        "test_samples": len(test_data),
        "calibration_samples": int(calibration.shape[0]),
        "checkpoint": str(checkpoint),
        "final_training_loss": loss,
        "training_seconds": training_seconds,
        "fp32_accuracy": fp32_accuracy,
        "generated_c_int8_accuracy": int8_accuracy,
        "accuracy_drop_percentage_points": (fp32_accuracy - int8_accuracy) * 100.0,
        "python_c_compared_bytes": compared_bytes,
        "python_c_mismatched_bytes": mismatches,
        "ptq_compile_seconds": ptq_seconds,
        "generated_c_test_seconds": c_seconds,
        "portable_memory": portable.memory_report.json_data()["compile_time"],
        "esp32s3_memory": esp32s3.memory_report.json_data()["compile_time"],
        "esp32s3_optimized_steps": optimized,
        "esp_idf_project": str(project.root),
    }
    (arguments.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print("\nBoardless build:")
    print(f"  cd {project.root}")
    print("  idf.py set-target esp32s3")
    print("  idf.py build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
