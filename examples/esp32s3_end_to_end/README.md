# FP32 PyTorch to ESP32-S3 demo

This example demonstrates the complete BakeNN deployment path in one command:

```text
PyTorch FP32 eval model
  -> torch.export capture
  -> representative-data PTQ
  -> typed INT8 graph and static SRAM plan
  -> ESP-NN/portable kernel selection
  -> standalone C and self-contained ESP-IDF project
```

The demo model is a deterministic, untrained RGB CNN with Conv2D, Depthwise
Conv2D, 1x1 Conv2D, global average pooling and FullyConnected. It is intended
to demonstrate compilation, not classification accuracy. For trained-model
results, use the MNIST and CIFAR-10 examples.

## Generate

```bash
python -m pip install -e '.[torch]'
python examples/esp32s3_end_to_end/generate.py \
  --output build/esp32s3_end_to_end
```

The command creates:

- `generated/`: model C, weights, selected kernels and manifest;
- `esp_idf/`: a self-contained ESP-IDF application and pinned ESP-NN sources;
- `demo_summary.json`: selected kernel IDs and static resource summary.

No TFLM runtime, FlatBuffer parser or target-side model loader is required.

## Build without a board

With ESP-IDF 5.5 activated:

```bash
cd build/esp32s3_end_to_end/esp_idf
idf.py set-target esp32s3
idf.py build
idf.py size-components
```

## Flash and measure

```bash
idf.py -p /dev/ttyUSB0 flash monitor
```

The generated runner initializes the input at its affine zero point, executes
one cold call, 8 warmups and 101 measured calls, and prints exact median/p95
cycle counts, the model arena, FreeRTOS stack high-water mark, output FNV-1a
checksum and output bytes. A physical board is required for meaningful cycle,
cache and energy measurements.
