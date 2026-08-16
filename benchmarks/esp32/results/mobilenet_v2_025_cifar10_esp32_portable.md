# MobileNetV2-0.25 with BakeNN portable C on a physical ESP32

The exact model, quantized graph, input, board, clock, compiler, warmups, and
measured-run count used by the ESP-NN experiment were held constant. Only the
BakeNN kernel policy changed from ESP-NN to portable C.

| Metric | BakeNN portable C | BakeNN + ESP-NN |
|---|---:|---:|
| Median latency | 68,531,150 cycles / 285.546 ms | 23,444,506 cycles / 97.685 ms |
| Throughput | 3.50 inference/s | 10.24 inference/s |
| App binary | 459,088 bytes | 465,296 bytes |
| Flash code | 64,080 bytes | 70,296 bytes |
| DRAM | 31,900 bytes | 31,900 bytes |
| Python INT8 vs board | 0 / 10 mismatches | 0 / 10 mismatches |

On this workload ESP-NN was **2.923x faster** than portable C, at the cost of
6,208 additional app-binary bytes. Both paths returned exactly the same ten
INT8 output bytes as BakeNN's Python integer reference.

The raw values, hashes, input contract, and limitations are recorded in
[`mobilenet_v2_025_cifar10_esp32_portable.json`](mobilenet_v2_025_cifar10_esp32_portable.json).
