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

from evidence_utils import (  # noqa: E402
    artifact_set_sha256,
    corpus_sha256,
    file_record,
    logical_checkpoint_sha256,
    sha256_file,
)


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
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=REPOSITORY / "examples/mnist/evidence",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="load an existing FP32 state_dict and skip training",
    )
    parser.add_argument("--calibration-per-class", type=int, default=16)
    parser.add_argument("--physical-per-class", type=int, default=10)
    parser.add_argument("--reference-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    return parser.parse_args()


def source_state() -> tuple[str, bool]:
    """Capture provenance before this command writes any generated evidence."""

    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


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


def balanced_corpus(
    dataset: datasets.MNIST,
    per_class: int,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    selected_images: list[torch.Tensor] = []
    selected_labels: list[int] = []
    counts = [0] * 10
    for index, label in enumerate(dataset.targets):
        digit = int(label)
        if counts[digit] < per_class:
            selected_images.append(dataset.data[index])
            selected_labels.append(digit)
            counts[digit] += 1
        if all(count == per_class for count in counts):
            break
    if not selected_images or not all(count == per_class for count in counts):
        raise RuntimeError("could not construct class-balanced MNIST corpus")
    raw_images = torch.stack(selected_images).numpy().astype(np.uint8, copy=False)
    labels = np.asarray(selected_labels, dtype=np.uint8)
    fp32 = torch.from_numpy(raw_images.copy()).unsqueeze(1).to(torch.float32) / 255.0
    return fp32, raw_images, labels


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
    source_commit, source_working_tree_dirty = source_state()
    if arguments.checkpoint is None and arguments.epochs <= 0:
        raise ValueError("epochs must be positive when training a checkpoint")
    if arguments.calibration_per_class <= 0 or arguments.physical_per_class <= 0:
        raise ValueError("calibration-per-class and physical-per-class must be positive")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    arguments.evidence_dir.mkdir(parents=True, exist_ok=True)
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
    training_performed = arguments.checkpoint is None
    if arguments.checkpoint is None:
        train(model, training_loader, arguments.epochs)
    else:
        state = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"loaded FP32 checkpoint: {arguments.checkpoint}", flush=True)
    model.eval()
    checkpoint_path = arguments.output_dir / "mnist_fp32.pt"
    torch.save(model.state_dict(), checkpoint_path)
    evidence_checkpoint = arguments.evidence_dir / "mnist_fp32.pt"
    shutil.copy2(checkpoint_path, evidence_checkpoint)
    checkpoint_logical_hash = logical_checkpoint_sha256(model.state_dict())
    fp32_accuracy = accuracy(model, test_loader)
    print(f"FP32 test accuracy: {fp32_accuracy * 100.0:.2f}%", flush=True)

    calibration, calibration_raw, calibration_labels = balanced_corpus(
        training_set, arguments.calibration_per_class
    )
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

    physical_fp32, physical_raw, physical_labels = balanced_corpus(
        test_set, arguments.physical_per_class
    )
    physical_codes = quantize_mnist_corpus(compiled.plan, physical_raw)
    physical_outputs = np.concatenate(
        [
            bakenn.run_reference(compiled.plan, physical_codes[index : index + 1])
            for index in range(len(physical_codes))
        ],
        axis=0,
    ).reshape(len(physical_codes), output_size)
    physical_accuracy = float(
        np.mean(np.argmax(physical_outputs, axis=1) == physical_labels)
    )
    np.testing.assert_array_equal(
        physical_codes[:1],
        bakenn.quantize_input(
            compiled.plan,
            physical_fp32[:1].numpy().transpose(0, 2, 3, 1),
        ),
    )

    manifest = json.loads(compiled.artifacts.manifest.read_text(encoding="utf-8"))
    report = {
        "schema_version": 2,
        "seed": arguments.seed,
        "epochs": arguments.epochs if training_performed else 0,
        "training_performed": training_performed,
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
        "checkpoint_file_sha256": sha256_file(checkpoint_path),
        "checkpoint_logical_sha256": checkpoint_logical_hash,
        "calibration_corpus_sha256": corpus_sha256(
            calibration_raw, calibration_labels, domain="calibration-u8"
        ),
        "physical_corpus_samples": len(physical_codes),
        "physical_corpus_reference_accuracy": physical_accuracy,
    }
    report_path = arguments.output_dir / "mnist_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    evidence_payloads = {
        "calibration_images_u8.bin": calibration_raw.tobytes(order="C"),
        "calibration_labels_u8.bin": calibration_labels.tobytes(order="C"),
        "physical_test_images_u8.bin": physical_raw.tobytes(order="C"),
        "physical_test_labels_u8.bin": physical_labels.tobytes(order="C"),
        "physical_test_inputs_int8.bin": physical_codes.tobytes(order="C"),
        "physical_expected_outputs_int8.bin": physical_outputs.tobytes(order="C"),
    }
    evidence_files: list[dict[str, object]] = [
        file_record(
            evidence_checkpoint,
            root=arguments.evidence_dir,
            role="serialized FP32 state_dict checkpoint",
        )
    ]
    for name, data in evidence_payloads.items():
        path = arguments.evidence_dir / name
        path.write_bytes(data)
        evidence_files.append(
            file_record(path, root=arguments.evidence_dir, role=name.removesuffix(".bin"))
        )

    generated_evidence = arguments.evidence_dir / "generated"
    generated_evidence.mkdir(parents=True, exist_ok=True)
    generated_sources = [
        compiled.artifacts.header,
        compiled.artifacts.model_source,
        compiled.artifacts.weights_header,
        compiled.artifacts.weights_source,
        compiled.artifacts.kernels_header,
        compiled.artifacts.kernels_source,
        compiled.artifacts.manifest,
        compiled.artifacts.memory_report_text,
        compiled.artifacts.memory_report_json,
    ]
    copied_generated: list[Path] = []
    for source in generated_sources:
        destination = generated_evidence / source.name
        shutil.copy2(source, destination)
        copied_generated.append(destination)
        evidence_files.append(
            file_record(
                destination,
                root=arguments.evidence_dir,
                role="generated deployment artifact",
            )
        )
    evidence_report = arguments.evidence_dir / "mnist_report.json"
    shutil.copy2(report_path, evidence_report)
    evidence_files.append(
        file_record(
            evidence_report,
            root=arguments.evidence_dir,
            role="host training, PTQ and accuracy report",
        )
    )

    dataset_sources: list[dict[str, object]] = []
    for source in sorted((arguments.data_dir / "MNIST/raw").glob("*-ubyte")):
        dataset_sources.append(
            {
                "name": source.name,
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )
    evidence = {
        "schema_version": 1,
        "scope": "trained FP32 MNIST checkpoint -> PTQ -> standalone C11",
        "source_commit": source_commit,
        "working_tree_dirty": source_working_tree_dirty,
        "software": {
            "bakenn": bakenn.__version__,
            "torch": torch.__version__,
            "torchvision": getattr(sys.modules.get("torchvision"), "__version__", None),
            "compiler": arguments.cc,
        },
        "training": {
            "performed": training_performed,
            "epochs": arguments.epochs if training_performed else 0,
            "seed": arguments.seed,
            "training_samples": len(training_set),
            "fp32_test_accuracy": fp32_accuracy,
        },
        "checkpoint": {
            "file_sha256": sha256_file(evidence_checkpoint),
            "logical_tensor_sha256": checkpoint_logical_hash,
        },
        "calibration": {
            "selection": "first N samples per class in canonical MNIST training order",
            "samples_per_class": arguments.calibration_per_class,
            "samples": len(calibration),
            "shape_nhw": list(calibration_raw.shape),
            "corpus_sha256": corpus_sha256(
                calibration_raw, calibration_labels, domain="calibration-u8"
            ),
        },
        "physical_test_corpus": {
            "selection": "first N samples per class in canonical MNIST test order",
            "samples_per_class": arguments.physical_per_class,
            "samples": len(physical_codes),
            "input_shape_nhwc": list(physical_codes.shape),
            "raw_corpus_sha256": corpus_sha256(
                physical_raw, physical_labels, domain="physical-raw-u8"
            ),
            "quantized_input_sha256": sha256_file(
                arguments.evidence_dir / "physical_test_inputs_int8.bin"
            ),
            "expected_output_sha256": sha256_file(
                arguments.evidence_dir / "physical_expected_outputs_int8.bin"
            ),
            "python_int8_accuracy": physical_accuracy,
        },
        "generated_artifacts": {
            "set_sha256": artifact_set_sha256(
                arguments.evidence_dir, copied_generated
            ),
            "python_c_mismatched_bytes": mismatched_bytes,
            "generated_c_test_accuracy": c_accuracy,
            "arena_bytes": manifest["arena_bytes"],
            "constant_bytes": manifest["constant_bytes"],
        },
        "dataset_sources": dataset_sources,
        "files": sorted(evidence_files, key=lambda item: str(item["path"])),
    }
    evidence_path = arguments.evidence_dir / "mnist_evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"report: {report_path}", flush=True)
    print(f"evidence: {evidence_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
