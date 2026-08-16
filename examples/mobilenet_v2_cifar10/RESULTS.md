# Recorded MobileNetV2-0.25 CIFAR-10 run

Recorded on 2026-08-16 with BakeNN 0.1.0, PyTorch 2.10.0,
torchvision 0.25.0 and seed `20260816`. The torchvision MobileNetV2 factory
used `width_mult=0.25`, `num_classes=10`, a static `1x3x32x32` input and one
complete epoch over all 50,000 CIFAR-10 training images on CPU.

| Measurement | Result |
|---|---:|
| Final training loss | 2.22446 |
| FP32 accuracy, 10,000 test images | 21.10% |
| Generated-C INT8 accuracy, 10,000 test images | 20.95% |
| FP32 minus INT8 | +0.15 percentage points |
| Python INT8 vs generated C | 160 bytes, 0 mismatches |
| Training time | 276.9 s |
| PTQ and portable-C generation | 11.0 s |
| Generated-C full-test host time | 3.2 s |

The absolute accuracy is intentionally not presented as a competitive
CIFAR-10 result: one epoch primarily validates that a real trained checkpoint,
BatchNorm folding, representative-data PTQ and the full MobileNetV2 graph
survive deployment. The 0.15 percentage-point delta is the measured result of
this run, not a guarantee for arbitrary checkpoints or calibration sets. The
byte comparison was repeated from the saved checkpoint on 16 deterministic
test samples after the full-test run.

## Static deployment resources

| Resource | Portable C | ESP32-S3 + ESP-NN |
|---|---:|---:|
| Semantic constants | 261,608 B | 261,608 B |
| Emitted constant payload | 304,312 B | 304,344 B |
| Activation arena | 16,384 B | 16,384 B |
| Selected-kernel scratch | 0 B | 102,464 B |
| Total model arena | 16,384 B | 118,848 B |
| Caller-owned input + output | 3,082 B | 3,082 B |
| Generated-model heap calls | 0 | 0 |

The ESP32-S3 backend selected 54 optimized ESP-NN steps across Conv2D,
DepthwiseConv2D, global average pooling and Linear. The larger ESP32-S3 arena
is caused by the maximum scratch request of the selected optimized kernels; it
is a speed/memory tradeoff rather than activation growth.

These are model-artifact values known at AOT compilation. The emitted constant
payload is not the final ELF Flash size, and the arena is not whole-firmware
peak SRAM. ESP-IDF, application globals, caller I/O and stacks require a final
target link; cycles and stack high-water require a physical board. No ESP32-S3
cycle claim is made from the 3.2-second host run.
