# MobileNetV2-0.25: BakeNN versus TFLM on a physical ESP32

The same trained checkpoint, calibration set, frozen INT8 tensors, all-zero
real input, 240 MHz clock, eight warmups, and 101 measured runs were used for
all three paths.  The TFLite FlatBuffer was independently executed with the
desktop LiteRT reference interpreter before being compiled with Espressif's
official TFLM 1.4.0 and ESP-NN 1.2.6 components.

| Path | Median latency | App binary | Flash data + code | DRAM | Output mismatch |
|---|---:|---:|---:|---:|---:|
| BakeNN portable C | 285.546 ms | 459,088 B | 393.5 KiB | 31,900 B | 0 / 10 B |
| BakeNN + ESP-NN | **97.685 ms** | **465,296 B** | **399.5 KiB** | **31,900 B** | 0 / 10 B |
| TFLM + ESP-NN | 98.891 ms | 665,504 B | 590.0 KiB | 95,332 B | 0 / 10 B |

For this artifact, direct BakeNN-to-ESP-NN lowering was 2.92x faster than
portable C and 1.23% faster by cycle ratio than TFLM+ESP-NN.  TFLM+ESP-NN used
200,208 more app-binary bytes and 63,432 more linked DRAM bytes than
BakeNN+ESP-NN.  TFLM reserved an 80 KiB tensor arena and reported 81,436 bytes
used; BakeNN's compile-time activation arena was 16 KiB.

All three board paths produced the exact BakeNN Python INT8 reference output:

```text
36 -16 98 54 63 17 30 5 0 -27
FNV-1a: 0x6cc8e3fb
```

The final TFLM board line was:

```text
TFLM target=esp32 cpu_mhz=240 first_cycles=23665218 median_cycles=23733948 p95_cycles=23733948 stack_high_water_words=14428 arena_reserved=81920 arena_used=81436 input_zero_point=-5 mismatches=0
```

During validation, the comparison runner exposed an important arena-lifetime
rule: TFLM input storage may be reused after `Invoke()`, so the input must be
written again before every invocation.  Once fixed, the desktop TFLite
reference, TFLM board result, BakeNN Python reference, portable C, and direct
ESP-NN output all agreed byte-for-byte.

These are measurements for one model and one original ESP32 board, not a claim
that the same ordering holds for every network or target.  Full hashes,
versions, memory sections, and limitations are in
[`mobilenet_v2_025_cifar10_esp32_tflm_espnn.json`](mobilenet_v2_025_cifar10_esp32_tflm_espnn.json).
