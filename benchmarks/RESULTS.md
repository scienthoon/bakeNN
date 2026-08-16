# BakeNN benchmark evidence

This page separates physical measurements from host differential tests and
boardless cross-build evidence. A green cross-build proves that the generated
project compiles and links for a target; it is not a latency measurement.

## Physical nRF52840 measurements

The frozen `32 -> 16 -> 4` INT8 FC workload used identical qparams, weights,
biases, input bytes and output semantics in all four images. It ran on the same
FIT IoT-LAB nRF52840DK at 64 MHz with 8 warmups and 101 measured calls.

| Build | Median cycles | Linked Flash | Linked static SRAM | Output |
|---|---:|---:|---:|---|
| BakeNN direct CMSIS-NN FC | **3,786** | 20,920 B | 8,540 B | byte-exact |
| TFLM + CMSIS-NN FC | 5,418 | 69,640 B | 11,040 B | byte-exact |
| BakeNN portable FC | 8,706 | **20,764 B** | 8,540 B | byte-exact |
| TFLM reference FC | 9,342 | 63,176 B | 11,008 B | byte-exact |

For this workload, BakeNN direct CMSIS-NN used 30.1% fewer cycles, 70.0% less
linked Flash and 22.6% less linked static SRAM than TFLM using the same
CMSIS-NN FC family. All output bytes had FNV-1a `0x910c1fe2`. See the
[full FC report](tflm_compare/results/iotlab_447626_direct_cmsis_fc.md) for
toolchain versions, hashes and limitations.

A separate `1x4x4x1 -> 1x4x4x2` Conv2D run compared BakeNN portable C with the
pinned TFLM reference Conv2D implementation:

| Build | Median cycles | Linked Flash | Linked static SRAM | Arena |
|---|---:|---:|---:|---:|
| BakeNN portable Conv2D | **24,610** | **20,332 B** | **8,160 B** | 0 B |
| TFLM reference Conv2D | 27,441 | 61,760 B | 10,624 B | 2,048 B reserved |

This is 10.3% fewer cycles, 67.1% less linked Flash and 23.2% less linked static
SRAM for that small reference-kernel workload. It is not a CMSIS-NN Conv2D
comparison. The complete contract is in the
[Conv2D report](tflm_compare/results/iotlab_447609_conv.md).

## ESP32 and ESP32-S3 evidence

ESP-NN 1.2.6 is pinned at revision
`c0876179f1cf4b4b9073b4f81cb65c8051ccb476`. The current evidence is:

| Target/path | Numerical evidence | Toolchain evidence | Physical cycles |
|---|---|---|---|
| ESP32 trained MobileNetV2-0.25, BakeNN portable | board output matches Python INT8 reference | ESP-IDF 5.5.4, Xtensa GCC 14.2.0 | 68,531,150 cycles / 285.546 ms median |
| ESP32 trained MobileNetV2-0.25, BakeNN + ESP-NN | board output matches Python INT8 reference | ESP-IDF 5.5.4, ESP-NN 1.2.6 | **23,444,506 cycles / 97.685 ms median** |
| ESP32 trained MobileNetV2-0.25, TFLM + ESP-NN | board output matches desktop LiteRT and BakeNN reference | TFLM 1.4.0, ESP-NN 1.2.6 | 23,733,948 cycles / 98.891 ms median |
| ESP32-S3 SIMD Conv/Depthwise/FC/Pool | 10,000 random and edge tensors per graph through the official ANSI oracle, zero byte mismatches | real Xtensa S3 sources compile and link in ESP-IDF 5.5 | unmeasured |
| ESP32-C3 portable fallback | shared Python-reference/portable-C differential suite | ESP-IDF 5.5 project builds and links | unmeasured |

All three original-ESP32 paths used the same frozen graph, real-zero INT8
input, 240 MHz clock, 8 warmups and 101 measured runs. BakeNN+ESP-NN was 1.22%
lower latency than TFLM+ESP-NN for this artifact, while its app binary was
465,296 B versus 665,504 B and linked DRAM was 31,900 B versus 95,332 B. The
[full three-way report](esp32/results/mobilenet_v2_025_cifar10_esp32_tflm_espnn.md)
records model, output bytes, toolchains, hashes and limitations.

The boardless target proof is GitHub Actions run
[`31897480108`](https://github.com/scienthoon/bakeNN/actions/runs/31897480108)
at commit `f1fd64f`: ESP32, ESP32-S3 and ESP32-C3 all built successfully. A
physical ESP32-S3 is still required before publishing S3 SIMD cycle, cache or
energy claims. The generated ESP-IDF runner already prints first/median/p95
cycles, output checksum and FreeRTOS stack watermark, so no compiler change is
needed for that measurement.

## Claim boundary

These results demonstrate that AOT execution can remove measurable runtime
overhead and reduce linked resources for the frozen workloads above. They do
not establish a universal speed ranking over TFLM, every model or every MCU.
New models and targets need the same-model, same-kernel-family, same-compiler
protocol in [the benchmark specification](tflm_compare/README.md).
