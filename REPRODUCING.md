# Reproducing BakeNN 0.1 results

This document separates host correctness tests, boardless target builds and
physical performance measurements.  A cross-build proves that an artifact
compiles and links; it is not a latency claim.

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
