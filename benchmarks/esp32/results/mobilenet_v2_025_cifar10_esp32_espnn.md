# MobileNetV2-0.25 on a physical original ESP32

On 2026-08-16, BakeNN compiled the one-epoch FP32 CIFAR-10 checkpoint into a
standalone ESP-IDF application using 53 bundled ESP-NN selections.  The app was
cross-compiled, flashed, and run on an ESP32-D0WDQ6 revision v1.1 at 240 MHz.

| Metric | Measured value |
|---|---:|
| Median latency | 23,444,506 cycles / 97.685 ms |
| p95 latency | 23,452,255 cycles / 97.718 ms |
| Throughput | 10.24 inference/s |
| App binary | 465,296 bytes |
| ELF Flash data + code | 409,120 bytes |
| DRAM | 31,900 bytes |
| BakeNN activation arena | 16,384 bytes |
| BakeNN scratch | 0 bytes |
| Python INT8 vs board | 10 bytes compared, 0 mismatches |

The board emitted:

```text
BAKENN target=esp32 cpu_mhz=240 first_cycles=23479356 median_cycles=23444506 p95_cycles=23452255 stack_high_water_words=6752 arena=16384
BAKENN_OUTPUT_FNV1A=0x6cc8e3fb
BAKENN_OUTPUT 36 -16 98 54 63 17 30 5 0 -27
```

The BakeNN Python integer reference independently produced the same ten bytes
and the same FNV-1a checksum.  This demonstrates cross-architecture byte-exact
execution for this artifact.  It does not compare against TFLM and does not by
itself establish that BakeNN is faster than TFLM on this board.

The full environment, hashes, memory sections, input contract, and limitations
are recorded in
[`mobilenet_v2_025_cifar10_esp32_espnn.json`](mobilenet_v2_025_cifar10_esp32_espnn.json).
