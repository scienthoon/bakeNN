# nRF52840 Conv2D comparison

This is a separate physical run from the MLP result. Both images implement the
same static `1x4x4x1 -> 1x4x4x2` 3x3 SAME Conv2D with fused ReLU, input bytes at
the input zero-point, 8 warmups, and 101 timed calls.

| Implementation | Median / p95 cycles | Flash | Linked SRAM | Arena |
|---|---:|---:|---:|---:|
| BakeNN portable C | 24,610 / 24,610 | 20,332B | 8,160B | 0B |
| TFLM reference Conv2D | 27,441 / 27,441 | 61,760B | 10,624B | 2,048B reserved / 516B reported used |

For this small Conv2D, BakeNN used 10.3% fewer cycles, 67.1% less linked Flash,
and 23.2% less linked SRAM reservation. This is still the Zephyr 2.7 reference
TFLM kernel, not CMSIS-NN; a CMSIS-NN comparison remains the important next
test before making a claim about large, Conv-heavy production models.

The host BakeNN reference predicts all 32 output codes are `1`, and the UART
prefixes from both images were `1`. The UART line was truncated before all 32
bytes were transmitted, so the machine-readable result intentionally leaves
byte-level output metrics unmeasured rather than claiming parity from a partial
line.

The generated Conv FlatBuffer is version 2 because the pinned Zephyr TFLM
resolver rejects versions 1 and 3. The input/model hashes and ELF hashes are in
the companion JSON.
