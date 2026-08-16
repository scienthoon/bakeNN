# Physical FC result: BakeNN direct CMSIS-NN versus TFLM

This result uses one fixed INT8 `32→16→4` MLP, identical tensor qparams,
weights, biases, fused hidden ReLU, raw input bytes, board, compiler, and cycle
protocol. All four firmware images ran on the same Nordic nRF52840DK node in
FIT IoT-LAB experiment `447626`.

The [validated JSON](iotlab_447626_direct_cmsis_fc.json),
[four-path CSV](iotlab_447626_direct_cmsis_fc.csv), and
[raw UART transcript](iotlab_447626_direct_cmsis_fc_uart.txt) preserve the
machine-readable result. This historical run used one input and a dirty
BakeNN worktree; the exact linked ELFs are preserved by SHA-256, but the result
must not be described as a clean `v0.1.0` multi-input measurement.

## Result

| implementation | FC kernel path | median cycles | p95 cycles | linked Flash (`text+data`) | linked SRAM (`data+bss`) | model arena | output |
|---|---|---:|---:|---:|---:|---:|---|
| BakeNN direct CMSIS-NN | generated call → CMSIS-NN v4 FC | **3,786** | 3,786 | 20,920 B | 8,540 B | 16 B | `[-5, -4, -4, -4]` |
| TFLM + CMSIS-NN | interpreter → TFLM wrapper → CMSIS-NN v4 FC | 5,418 | 5,418 | 69,640 B | 11,040 B | 1,024 B reserved / 580 B used | `[-5, -4, -4, -4]` |
| BakeNN portable | generated scalar C FC | 8,706 | 8,706 | **20,764 B** | 8,540 B | 16 B | `[-5, -4, -4, -4]` |
| TFLM reference | interpreter → reference FC | 9,342 | 9,342 | 63,176 B | 11,008 B | 1,024 B reserved | `[-5, -4, -4, -4]` |

The output FNV-1a checksum was `0x910c1fe2` in every image: there were zero
byte mismatches. On this small two-layer FC model, BakeNN's direct CMSIS-NN
path used 30.1% fewer cycles than TFLM+CMSIS-NN, 56.5% fewer than BakeNN
portable, and 59.5% fewer than TFLM reference. It was 156 B larger in linked
Flash than BakeNN portable because the model only pulls the two required
CMSIS-NN FC source files.

This is evidence for this fixed FC graph on this board, not a claim that every
model is 30.1% faster. Convolution, larger graphs, caches, different compilers,
and different MCUs require their own measurements. The SRAM column is the
linker's static reservation, not a simultaneous whole-firmware peak proof.

## Reproduction contract

- Board: `nrf52840dk-1.saclay.iot-lab.info`, nRF52840 Cortex-M4F at 64 MHz
- Zephyr: 2.7.5
- Compiler: Zephyr SDK 0.13.1, `arm-zephyr-eabi-gcc` 10.3.0, speed optimization
- TFLM revision: `9156d050927012da87079064db59d07f03b8baf6`
- CMSIS-NN: v4.0.0, revision `ca5dc34313be2ee5c46652917c30baac96c52621`
- BakeNN source base: `c948a87e007f1cd30e3dd698486fa85a7d7f7bd4` plus this direct-backend change
- TFLite FlatBuffer SHA-256: `c06327251b6a828493be40994e6378e7d3495ed3538a16f7d418f46aa35d8401`
- BakeNN generated weights SHA-256: `3ae5e6f8df9c0094aed6d98a9d89e9bd71d4c88400fb977fa99917cfc0a89653`
- Input: 32 INT8 bytes, all at input zero-point `-3`
- Input SHA-256: `a2c6cb39ea5a7ad89edf34c07004b85a05ceef6b1480c549da6b328b31def097`
- Protocol: 8 warmups, 101 timed calls, exact sorted median and p95

The direct backend originally measured 13,222 cycles because the freestanding
build emitted a function call to `memcpy` for every four-byte CMSIS-NN DSP
load. `BAKENN_CMSIS_NN_BUILTIN_MEMORY` now preserves those fixed-width loads as
compiler builtins; the resulting inner loop contains `SMLAD` and no `memcpy`
calls. This regression is covered by cross-link and disassembly tests.
