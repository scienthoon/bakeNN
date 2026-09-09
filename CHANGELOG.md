# Changelog

All notable changes are recorded here. BakeNN follows semantic versioning once
its public API stabilizes; pre-1.0 releases may still contain breaking changes.

## Unreleased

### Fixed

- preserve unique TFLite tensor/constant identities and float32 RELU6 rounding;
  accept constant-input RESHAPE without options and FC v6 with optional bias;
- reject unsupported MaxPool1d dilation/ceil semantics and shared-storage
  in-place mutations instead of silently changing the model;
- validate ReduceMean ranks/axes, extended-op target ABI limits, Slice qparams
  and unrepresentable bias scales before generation;
- preserve repeated Sequential calls, validate cross-layout channel order,
  align Flatten accuracy comparisons, and use wide fixed-point scalar math;
- correct singleton-axis bilinear corner alignment in FP32 evaluation,
  reference execution and generated C;
- reject unsafe ESP-NN padded 1x1 and vendor requantization cases, and enforce
  CMSIS-NN coordinate/reduction field bounds with portable fallback;
- preserve CMSIS DSP depthwise scratch bounds for fully padded vertical windows;
  enforce int16 origins on ESP-NN generic depthwise paths, including S3 fallbacks;
- retry an SRAM-feasible kernel combination when a preferred shared-scratch
  envelope exceeds the declared target budget;
- publish ESP-IDF/Zephyr exports transactionally without overwriting existing
  application files or modifying source artifact trees;
- preserve measured optimization flags during freestanding builds and reject
  conflicting build overrides.

### Validation

- add 149 regression/control cases and connect optional frontend and cross-build
  cases to their dependency-specific CI jobs; local full suite: 504 tests and
  6 subtests passed;
- document audit scope, reproduced backend failures, remaining issues, and follow-up
  priorities in `docs/reviews/2026-09-09/AUDIT.md`.

## 1.0.0 - 2026-08-26

### Added

- strict optional host-side import of supported fully-quantized static INT8
  TFLite models into the existing typed IR;
- a separately versioned TFLite raw-code AveragePool profile, gated byte for
  byte against LiteRT's built-in reference implementation;
- versioned public C/C++ firmware ABI and manifest validation;
- transactional generated-artifact publication with content inventories and
  provenance hashes, including manifest-payload self-integrity;
- explicit measured versus static-priority kernel-selection policies;
- 1.0 release gates for the supported Python, PyTorch, model and target matrix.

### Changed

- backend selection no longer changes PTQ weight granularity implicitly;
- vendored CMSIS-NN compatibility changes now carry per-file notices and exact
  upstream/patched provenance;
- stable APIs, numerical profiles and generated artifact schemas follow the
  compatibility policy in `STABILITY.md`.

## 0.1.0 - 2026-08-16

### Added

- static typed INT8 IR, verifier, liveness planner and heap-free C11 backend;
- PyTorch `torch.export` FP32 capture and representative-data PTQ;
- Conv2D, DepthwiseConv2D, Linear, pooling, elementwise, reshape, concatenate,
  softmax and the documented extended vision/audio operator surface;
- Python integer reference and generated-C byte-exact differential tests;
- direct CMSIS-NN FC/Conv/Depthwise/Pool backend for Cortex-M4 DSP targets;
- direct ESP-NN Conv/Depthwise/FC/Pool backend for ESP32-S3 and optimized
  Conv/Depthwise support for ESP32, with ESP32-C3 portable fallback;
- self-contained Zephyr, freestanding GNU and ESP-IDF target projects;
- physical nRF52840 BakeNN-versus-TFLM benchmark reports;
- deterministic FP32-to-ESP32-S3 end-to-end demo;
- deterministic text and JSON memory reports with activation lifetimes, arena
  reuse, selected-kernel scratch, target-budget headroom and explicit
  post-link/physical-measurement boundaries;
- trained MobileNetV2 physical ESP32 measurements for portable C, direct
  ESP-NN and TFLM+ESP-NN with byte-exact output evidence;
- deterministic release-evidence archives, PyPI Trusted Publishing workflow,
  stability policy, third-party notices and reproduction guide.

### Intentional constraints

- fixed static shapes, batch one and one public input/output;
- compile-time model replacement rather than a runtime model loader;
- no target-side float fallback or heap allocation;
- narrower fail-closed operator surface than TFLM.
