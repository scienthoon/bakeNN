# Reproducing BakeNN 0.1 results

This document separates host correctness tests, boardless target builds and
physical performance measurements.  A cross-build proves that an artifact
compiles and links; it is not a latency claim.

The two evidence classes have separate indexes:

- [`benchmarks/physical/`](benchmarks/physical/README.md) contains only runs
  with a physical-board UART transcript;
- [`benchmarks/cross_build/`](benchmarks/cross_build/README.md) contains
  boardless compiler/linker evidence with all runtime metrics unmeasured.

## Host environment and full test suite

```bash
git clone https://github.com/scienthoon/bakeNN.git
cd bakeNN
git checkout v0.1.0
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install ".[test,torch]"
python -m pytest -q
```

The generated-C tests invoke GCC or Clang, use strict C11 warnings and run
ASan/UBSan where supported.  CI additionally cross-links Cortex-M0+, Cortex-M4
and RV32IMC artifacts and builds ESP-IDF projects for ESP32, ESP32-S3 and
ESP32-C3.

## Build and inspect release distributions

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
shasum -a 256 dist/*.whl dist/*.tar.gz
```

Install the wheel into a fresh environment outside the checkout, import
`bakenn`, assert `bakenn.__version__ == "0.1.0"`, compile the smoke graph, and
compile the emitted C with the host compiler.  The release workflow performs
the same build once, publishes those exact files to PyPI through OIDC, and
attaches them and their checksums to the GitHub release.

## Build the deterministic evidence archive

```bash
python scripts/build_release_evidence.py \
  --output dist/bakenn-0.1.0-evidence.zip
```

The archive contains checked-in JSON/Markdown/UART evidence, the benchmark
protocol, this reproduction guide and a manifest tying the archive to its git
commit.  A release operator may additionally include exact local ELF/map files:

```bash
python scripts/build_release_evidence.py \
  --output dist/bakenn-0.1.0-evidence.zip \
  --artifact nrf52840-bakenn-cmsis.elf=/path/to/zephyr.elf \
  --artifact nrf52840-bakenn-cmsis.map=/path/to/zephyr.map
```

An absent physical artifact is reported as absent; the script never invents a
measurement or substitutes a host result.

## Trained MNIST evidence and replay

The four-epoch checkpoint, exact calibration bytes, generated C and a
class-balanced 100-image board corpus are committed under
`examples/mnist/evidence/`. Verify the recorded payload hashes:

```bash
python - <<'PY'
import hashlib, json, pathlib
root = pathlib.Path("examples/mnist/evidence")
manifest = json.loads((root / "mnist_evidence.json").read_text())
for item in manifest["files"]:
    actual = hashlib.sha256((root / item["path"]).read_bytes()).hexdigest()
    assert actual == item["sha256"], (item["path"], actual)
print("MNIST evidence payload hashes: PASS")
PY
```

Replay PTQ and generated-C evaluation from the frozen checkpoint:

```bash
PYTHONPATH=src python examples/mnist/run_mnist.py \
  --checkpoint examples/mnist/evidence/mnist_fp32.pt
```

The replay still downloads/loads the canonical MNIST files so that FP32 and C
accuracy are recomputed over all 10,000 test images. Their extracted IDX
SHA-256 values are recorded in `mnist_evidence.json`.

Generate and cross-build the original-ESP32 full-model firmware:

```bash
PYTHONPATH=src python examples/mnist/generate_esp32_benchmark.py
cd build/mnist_esp32_physical/esp_idf
. "$IDF_PATH/export.sh"
idf.py set-target esp32
idf.py build
idf.py size
```

This is still cross-build evidence. It becomes physical evidence only after
`idf.py -p PORT flash monitor` produces a transcript containing 100 samples,
99 correct classifications, 1,000 compared bytes, zero mismatches, output
FNV-1a `0x55fb9e60`, cycles and the four expected provenance hashes.

## microTVM AOT+USMP+CMSIS-NN baseline

The trained MNIST graph is also compared with the official Apache TVM 0.16.0
source release under a shared quantized contract. Build TVM with
`USE_MICRO=ON`, `USE_CMSISNN=ON` and `USE_LLVM=OFF`, then run:

```bash
PYTHONPATH="$TVM_SOURCE/python:src:examples/mnist:benchmarks/tflm_compare" \
TVM_LIBRARY_PATH="$TVM_BUILD" TOPHUB_LOCATION=NONE \
python benchmarks/microtvm_compare/build_mnist.py \
  --tvm-source "$TVM_SOURCE" \
  --tvm-build "$TVM_BUILD"
```

The script validates the checkpoint/calibration hashes, emits the common
`.tflite`, partitions five operators to CMSIS-NN, performs a 1,000-byte host C
differential, and cross-links both freestanding Cortex-M4 ELFs. Exact source
archive hash, commands, generated sources and result fields are documented in
[`benchmarks/microtvm_compare/`](benchmarks/microtvm_compare/README.md).
Cross-linked Flash/SRAM are not physical cycle evidence.

## Physical nRF52840 evidence

The frozen `32 -> 16 -> 4` FC comparison is documented in
`benchmarks/tflm_compare/results/iotlab_447626_direct_cmsis_fc.md`.  Its raw
UART, exact model/input hashes, Zephyr/TFLM/CMSIS-NN revisions, cycles and
linked Flash/SRAM are stored beside it.  The historical run used FIT IoT-LAB
experiment 447626 and one real-zero INT8 input; it must not be described as a
multi-input accuracy or energy measurement.

## Physical original-ESP32 evidence

The trained MobileNetV2-0.25 comparison is recorded in
`benchmarks/esp32/results/mobilenet_v2_025_cifar10_esp32_tflm_espnn.json`.
It compares BakeNN portable C, direct BakeNN-to-ESP-NN and TFLM+ESP-NN on the
same ESP32-D0WDQ6 at 240 MHz.  The `.tflite` was serialized from the frozen
BakeNN graph and independently checked with LiteRT's reference interpreter.
All board paths produced the same ten INT8 output bytes.

The timed input is the affine real-zero tensor.  The result therefore measures
the fixed graph and kernel paths, not camera preprocessing or energy.  Use the
JSON hashes and UART transcript to identify the exact artifacts.  Do not
generalize the measured ranking to ESP32-S3, another model or another compiler.

## Benchmark claim rules

- Freeze and hash the model and raw input corpus.
- Use the same board, clock, compiler settings, warmups and run count.
- Compare raw output bytes and report mismatches.
- Preserve UART, ELF, map/section evidence and versioned JSON.
- Mark missing initialization, stack, energy or whole-firmware peak metrics as
  unmeasured.
- Never use host latency or a boardless cross-build as MCU performance.

The full metric schema and memory-accounting rules are in
`benchmarks/tflm_compare/README.md`.
