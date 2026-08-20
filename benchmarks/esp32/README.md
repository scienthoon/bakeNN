# ESP32 physical-board results

This directory contains raw, reproducible BakeNN measurements from physical
ESP32 boards.  A result is evidence for the exact model, generated artifact,
toolchain, configuration, and input stated in its JSON file; it is not a claim
about every ESP32 board or every model.

The MobileNetV2 result was produced by compiling a trained FP32 PyTorch model
with BakeNN PTQ, selecting the bundled ESP-NN kernels, cross-linking an ESP-IDF
application, flashing an original ESP32, and comparing the UART output with the
BakeNN Python INT8 reference.

The [trained MNIST full-model result](results/mnist_trained_esp32.md) runs 100
frozen class-balanced inputs on the board, matches all 1,000 Python reference
output bytes, and records cycles, linked Flash/SRAM, stack watermark, raw UART,
raw size-command output and complete artifact hashes.

The same frozen graph was also measured through BakeNN portable C and the
official Espressif TFLM + ESP-NN components.  All three board outputs matched
the BakeNN Python reference byte-for-byte.  See the
[three-way result](results/mobilenet_v2_025_cifar10_esp32_tflm_espnn.md) and its
[machine-readable evidence](results/mobilenet_v2_025_cifar10_esp32_tflm_espnn.json).
