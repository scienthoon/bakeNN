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
- [Original ESP32 trained MNIST full model](../esp32/results/mnist_trained_esp32.md):
  100 frozen inputs, byte-exact Python/board outputs, cycles, Flash/SRAM and
  stack watermark.

## Trained MNIST full-model result

The repository freezes a trained FP32 checkpoint, calibration corpus, 100
class-balanced physical inputs and expected INT8 outputs under
[`examples/mnist/evidence`](../../examples/mnist/evidence/README.md). The
[physical ESP32 result](../esp32/results/mnist_trained_esp32.md) includes the
[raw UART transcript](../esp32/results/mnist_trained_esp32_uart.txt), complete
hash provenance and linked-memory measurements.

The physical acceptance line reports:

```text
samples=100 correct=99 compared_bytes=1000 mismatches=0
```

plus first/median/p95 cycles, output FNV-1a `0x55fb9e60` and all four
provenance hashes.
