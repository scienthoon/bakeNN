# Trained MobileNetV2-0.25 on ESP32-S3

This example is the representative full-model BakeNN path. It trains an
unmodified torchvision MobileNetV2 topology (`width_mult=0.25`, 10 classes) on
CIFAR-10 at 32x32, performs representative-data PTQ, runs the generated C over
the test set, checks Python INT8 and C output bytes, and emits a self-contained
ESP32-S3 ESP-IDF project.

It is deliberately an MCU-sized MobileNetV2, not the 224x224 ImageNet
`width_mult=1.0` model. One epoch validates the complete pipeline; it is not a
competitive CIFAR-10 training recipe.

## Dataset

The default path uses the same CIFAR-10 ImageFolder mirror as the training
matrix:

```bash
mkdir -p /private/tmp/bakenn-datasets/cifar10-fastai
curl -L -o /private/tmp/bakenn-datasets/cifar10-fastai.tgz \
  https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz
tar -xzf /private/tmp/bakenn-datasets/cifar10-fastai.tgz \
  -C /private/tmp/bakenn-datasets/cifar10-fastai --strip-components=1
```

## Train, quantize, and generate ESP-IDF

```bash
PYTHONPATH=src python examples/mobilenet_v2_cifar10/run.py
```

The default run uses all 50,000 training and 10,000 test images for one epoch,
with 20 class-balanced calibration images. It defaults to one PyTorch CPU
thread because that is materially faster on the recorded macOS host; use
`--threads` to tune another machine. Outputs are written to
`build/mobilenet_v2_cifar10`:

- `mobilenet_v2_025_cifar10_fp32.pt`: trained FP32 state dictionary;
- `generated_portable/`: host-verifiable standalone C;
- `generated_esp32s3/`: ESP-NN-selected model C and memory report;
- `esp_idf/`: self-contained ESP-IDF application;
- `report.json`: FP32/INT8 accuracy, byte agreement, timing and static memory.

For a quick pipeline smoke before the full run:

```bash
PYTHONPATH=src python examples/mobilenet_v2_cifar10/run.py \
  --max-train-samples 512 --max-test-samples 32 --calibration-per-class 1
```

## Build without a board

After activating ESP-IDF 5.5:

```bash
cd build/mobilenet_v2_cifar10/esp_idf
idf.py set-target esp32s3
idf.py build
idf.py size-components
```

The generated runner reports cold, median and p95 cycles, stack high-water
mark, model arena size, output checksum and raw output bytes when flashed. The
host timings in `report.json` are not MCU performance measurements.

The checked-in result from the default one-epoch run is in
[`RESULTS.md`](RESULTS.md).
