#!/usr/bin/env python3
"""Build the trained MNIST full-model physical benchmark for an original ESP32."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bakenn  # noqa: E402
from bakenn.targets import ESP32  # noqa: E402
from evidence_utils import (  # noqa: E402
    corpus_sha256,
    logical_checkpoint_sha256,
    sha256_file,
)
from run_mnist import MNISTNet, quantize_mnist_corpus  # noqa: E402


def _fnv1a(data: bytes) -> int:
    value = 2166136261
    for item in data:
        value ^= item
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def _main_source(symbol: str, contract: dict[str, object]) -> str:
    macro = symbol.upper()
    samples = int(contract["samples"])
    checkpoint = str(contract["checkpoint_logical_sha256"])
    calibration = str(contract["calibration_corpus_sha256"])
    corpus = str(contract["input_sha256"])
    expected = str(contract["expected_output_sha256"])
    return f'''#include "{symbol}.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "esp_cpu.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

extern const uint8_t corpus_bin_start[] asm("_binary_corpus_bin_start");
extern const uint8_t corpus_bin_end[] asm("_binary_corpus_bin_end");
extern const uint8_t labels_bin_start[] asm("_binary_labels_bin_start");
extern const uint8_t labels_bin_end[] asm("_binary_labels_bin_end");
extern const uint8_t expected_bin_start[] asm("_binary_expected_bin_start");
extern const uint8_t expected_bin_end[] asm("_binary_expected_bin_end");

#define SAMPLE_COUNT {samples}u
#define ARENA_STORAGE_SIZE ({macro}_ARENA_SIZE == 0u ? 1u : {macro}_ARENA_SIZE)

_Alignas({macro}_ARENA_ALIGNMENT)
static uint8_t model_arena[ARENA_STORAGE_SIZE];
static int8_t model_input[{macro}_INPUT_SIZE];
static int8_t model_output[{macro}_OUTPUT_SIZE];
static uint32_t measured_cycles[101];

static void sort_cycles(void) {{
    for (uint32_t index = 1u; index < 101u; ++index) {{
        const uint32_t value = measured_cycles[index];
        uint32_t position = index;
        while (position > 0u && measured_cycles[position - 1u] > value) {{
            measured_cycles[position] = measured_cycles[position - 1u];
            --position;
        }}
        measured_cycles[position] = value;
    }}
}}

static uint32_t fnv1a_update(uint32_t hash, const int8_t *data, size_t size) {{
    for (size_t index = 0u; index < size; ++index) {{
        hash ^= (uint8_t)data[index];
        hash *= UINT32_C(16777619);
    }}
    return hash;
}}

static uint32_t argmax_output(void) {{
    uint32_t best = 0u;
    for (uint32_t index = 1u; index < {macro}_OUTPUT_SIZE; ++index) {{
        if (model_output[index] > model_output[best]) {{
            best = index;
        }}
    }}
    return best;
}}

void app_main(void) {{
    const size_t corpus_bytes = (size_t)(corpus_bin_end - corpus_bin_start);
    const size_t label_bytes = (size_t)(labels_bin_end - labels_bin_start);
    const size_t expected_bytes = (size_t)(expected_bin_end - expected_bin_start);
    if (corpus_bytes != SAMPLE_COUNT * {macro}_INPUT_BYTES ||
        label_bytes != SAMPLE_COUNT ||
        expected_bytes != SAMPLE_COUNT * {macro}_OUTPUT_BYTES) {{
        printf("BAKENN_MNIST_ERROR corpus=%u labels=%u expected=%u\\n",
               (unsigned)corpus_bytes, (unsigned)label_bytes,
               (unsigned)expected_bytes);
        return;
    }}

    uint8_t *arena = {macro}_ARENA_SIZE == 0u ? NULL : model_arena;
    memcpy(model_input, corpus_bin_start, {macro}_INPUT_BYTES);
    uint32_t start = esp_cpu_get_cycle_count();
    {symbol}_infer(arena, model_input, model_output);
    const uint32_t first_cycles = esp_cpu_get_cycle_count() - start;
    for (uint32_t run = 0u; run < 8u; ++run) {{
        {symbol}_infer(arena, model_input, model_output);
        vTaskDelay(1);
    }}
    for (uint32_t run = 0u; run < 101u; ++run) {{
        start = esp_cpu_get_cycle_count();
        {symbol}_infer(arena, model_input, model_output);
        measured_cycles[run] = esp_cpu_get_cycle_count() - start;
        vTaskDelay(1);
    }}
    sort_cycles();

    uint32_t correct = 0u;
    uint32_t mismatches = 0u;
    uint32_t output_hash = UINT32_C(2166136261);
    for (uint32_t sample = 0u; sample < SAMPLE_COUNT; ++sample) {{
        memcpy(model_input,
               corpus_bin_start + sample * {macro}_INPUT_BYTES,
               {macro}_INPUT_BYTES);
        {symbol}_infer(arena, model_input, model_output);
        const int8_t *expected_output =
            (const int8_t *)(expected_bin_start + sample * {macro}_OUTPUT_BYTES);
        for (uint32_t index = 0u; index < {macro}_OUTPUT_SIZE; ++index) {{
            if (model_output[index] != expected_output[index]) {{
                ++mismatches;
            }}
        }}
        correct += argmax_output() == labels_bin_start[sample] ? 1u : 0u;
        output_hash = fnv1a_update(
            output_hash, model_output, {macro}_OUTPUT_BYTES);
        vTaskDelay(1);
    }}

    const UBaseType_t stack_words_free = uxTaskGetStackHighWaterMark(NULL);
    printf("BAKENN_MNIST target=%s cpu_mhz=%u samples=%u correct=%" PRIu32
           " accuracy_bp=%" PRIu32 " compared_bytes=%u mismatches=%" PRIu32
           " first_cycles=%" PRIu32 " median_cycles=%" PRIu32
           " p95_cycles=%" PRIu32 " stack_high_water_words=%" PRIu32
           " arena=%u\\n",
           CONFIG_IDF_TARGET, (unsigned)CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ,
           (unsigned)SAMPLE_COUNT, correct,
           (correct * UINT32_C(10000)) / SAMPLE_COUNT,
           (unsigned)(SAMPLE_COUNT * {macro}_OUTPUT_BYTES), mismatches,
           first_cycles, measured_cycles[50], measured_cycles[95],
           (uint32_t)stack_words_free, (unsigned){macro}_ARENA_SIZE);
    printf("BAKENN_MNIST_OUTPUT_FNV1A=0x%08" PRIx32 "\\n", output_hash);
    printf("BAKENN_MNIST_PROVENANCE checkpoint={checkpoint} calibration={calibration} "
           "input={corpus} expected={expected}\\n");
}}
'''


def generate(evidence_dir: Path, output: Path) -> dict[str, object]:
    evidence = json.loads((evidence_dir / "mnist_evidence.json").read_text())
    checkpoint = evidence_dir / "mnist_fp32.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    logical_hash = logical_checkpoint_sha256(state)
    if logical_hash != evidence["checkpoint"]["logical_tensor_sha256"]:
        raise RuntimeError("checkpoint logical tensor hash does not match evidence")

    calibration_shape = tuple(evidence["calibration"]["shape_nhw"])
    calibration_raw = np.fromfile(
        evidence_dir / "calibration_images_u8.bin", dtype=np.uint8
    ).reshape(calibration_shape)
    calibration_labels = np.fromfile(
        evidence_dir / "calibration_labels_u8.bin", dtype=np.uint8
    )
    calibration_hash = corpus_sha256(
        calibration_raw, calibration_labels, domain="calibration-u8"
    )
    if calibration_hash != evidence["calibration"]["corpus_sha256"]:
        raise RuntimeError("calibration corpus hash does not match evidence")
    calibration = (
        torch.from_numpy(calibration_raw.copy()).unsqueeze(1).to(torch.float32) / 255.0
    )

    model = MNISTNet().eval()
    model.load_state_dict(state)
    compiled = bakenn.compile_torch_ptq(
        model,
        calibration[:1],
        calibration,
        output / "generated",
        name="mnist_physical",
        target=ESP32,
        backend_options=bakenn.CBackendOptions(
            kernel_policy=bakenn.KernelPolicy.AUTO,
            enable_esp_nn=True,
            target=ESP32,
        ),
    )

    corpus_contract = evidence["physical_test_corpus"]
    input_shape = tuple(corpus_contract["input_shape_nhwc"])
    raw_shape = (input_shape[0], input_shape[1], input_shape[2])
    raw_images = np.fromfile(
        evidence_dir / "physical_test_images_u8.bin", dtype=np.uint8
    ).reshape(raw_shape)
    expected_inputs = np.fromfile(
        evidence_dir / "physical_test_inputs_int8.bin", dtype=np.int8
    ).reshape(input_shape)
    actual_inputs = quantize_mnist_corpus(compiled.plan, raw_images)
    np.testing.assert_array_equal(actual_inputs, expected_inputs)

    output_type = compiled.plan.tensors[compiled.plan.outputs[0]].tensor_type
    expected_outputs = np.fromfile(
        evidence_dir / "physical_expected_outputs_int8.bin", dtype=np.int8
    ).reshape(input_shape[0], output_type.numel)
    actual_outputs = np.concatenate(
        [
            bakenn.run_reference(compiled.plan, actual_inputs[index : index + 1])
            for index in range(input_shape[0])
        ],
        axis=0,
    ).reshape(expected_outputs.shape)
    np.testing.assert_array_equal(actual_outputs, expected_outputs)

    labels = np.fromfile(
        evidence_dir / "physical_test_labels_u8.bin", dtype=np.uint8
    )
    project = bakenn.export_esp_idf_project(
        compiled.artifacts, ESP32, output / "esp_idf"
    )
    main = project.main
    (main / "corpus.bin").write_bytes(actual_inputs.tobytes(order="C"))
    (main / "labels.bin").write_bytes(labels.tobytes(order="C"))
    (main / "expected.bin").write_bytes(expected_outputs.tobytes(order="C"))
    (main / "CMakeLists.txt").write_text(
        'idf_component_register(SRCS "main.c" INCLUDE_DIRS "." REQUIRES bakenn_model)\n'
        'target_add_binary_data(${COMPONENT_LIB} "corpus.bin" BINARY)\n'
        'target_add_binary_data(${COMPONENT_LIB} "labels.bin" BINARY)\n'
        'target_add_binary_data(${COMPONENT_LIB} "expected.bin" BINARY)\n',
        encoding="utf-8",
    )
    manifest = json.loads(compiled.artifacts.manifest.read_text())
    contract: dict[str, object] = {
        "schema_version": 1,
        "model": "trained MNISTNet full model",
        "samples": int(input_shape[0]),
        "checkpoint_file_sha256": sha256_file(checkpoint),
        "checkpoint_logical_sha256": logical_hash,
        "calibration_corpus_sha256": calibration_hash,
        "input_sha256": sha256_file(main / "corpus.bin"),
        "labels_sha256": sha256_file(main / "labels.bin"),
        "expected_output_sha256": sha256_file(main / "expected.bin"),
        "expected_output_fnv1a": f"0x{_fnv1a(expected_outputs.tobytes()):08x}",
        "expected_accuracy": float(
            np.mean(np.argmax(expected_outputs, axis=1) == labels)
        ),
        "target": ESP32.manifest(),
        "arena_bytes": manifest["arena_bytes"],
        "constant_bytes": manifest["constant_bytes"],
        "kernel_selections": [
            {
                "step": item.step_name,
                "kernel": item.kernel_id,
                "optimized": item.optimized,
                "reason": item.reason,
            }
            for item in compiled.artifacts.backend_plan.selections
        ],
    }
    (main / "main.c").write_text(
        _main_source(project.model_symbol, contract), encoding="utf-8"
    )
    (output / "benchmark_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=REPOSITORY / "examples/mnist/evidence",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "build/mnist_esp32_physical",
    )
    arguments = parser.parse_args()
    contract = generate(arguments.evidence_dir, arguments.output)
    print(json.dumps(contract, indent=2, sort_keys=True))
    print("\nBuild and flash:")
    print(f"  cd {arguments.output / 'esp_idf'}")
    print("  idf.py set-target esp32")
    print("  idf.py build")
    print("  idf.py -p /dev/cu.usbserial-210 flash monitor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
