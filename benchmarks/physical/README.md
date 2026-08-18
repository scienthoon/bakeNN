# Physical-board evidence

Every result indexed here was collected from a named physical board. Each
claim is scoped to its exact model, input corpus, compiler, firmware, clock and
kernel backend. Raw UART and machine-readable provenance are mandatory.

## Published measurements

- [nRF52840 four-way FC comparison](../tflm_compare/results/iotlab_447626_direct_cmsis_fc.md):
  BakeNN portable, BakeNN direct CMSIS-NN, TFLM reference and TFLM+CMSIS-NN.
- [nRF52840 Conv2D comparison](../tflm_compare/results/iotlab_447609_conv.md):
  BakeNN portable versus TFLM reference.
- [Original ESP32 trained MobileNetV2-0.25](../esp32/results/mobilenet_v2_025_cifar10_esp32_tflm_espnn.md):
  BakeNN portable, BakeNN+ESP-NN and TFLM+ESP-NN.

## Trained MNIST full-model gate

The repository now freezes a trained FP32 checkpoint, calibration corpus, 100
class-balanced physical inputs and expected INT8 outputs under
[`examples/mnist/evidence`](../../examples/mnist/evidence/README.md). The
original-ESP32 project builds successfully, but it is **not listed as a
physical result until a UART transcript from the board is checked in**.

The acceptance line must report:

```text
samples=100 correct=99 compared_bytes=1000 mismatches=0
```

plus first/median/p95 cycles, output FNV-1a and all four provenance hashes.

