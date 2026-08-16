# Changelog

All notable changes are recorded here. BakeNN follows semantic versioning once
its public API stabilizes; pre-1.0 releases may still contain breaking changes.

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
