# Physical CMSIS-NN result: BakeNN versus TFLM CMSIS-NN

> Historical preliminary run. Its BakeNN and TFLM output bytes differed, so it
> is not the final identical-semantics comparison. Use
> [`iotlab_447626_direct_cmsis_fc.md`](iotlab_447626_direct_cmsis_fc.md) for the
> corrected four-way FC result with zero byte mismatches.

This is a physical-target result from FIT IoT-LAB experiment `447626`, using
the same nRF52840DK node for both images.  The model and input bytes are the
same as the earlier MLP result; the TFLM image selects the pinned TFLM
`cmsis_nn/fully_connected.cc` wrapper and CMSIS-NN v4.0.0.

## Measured result

| implementation | backend | median cycles | p95 cycles | linked Flash | linked SRAM | arena | output |
|---|---|---:|---:|---:|---:|---:|---|
| BakeNN | generated portable C | 8,508 | 8,508 | 20,720 B | 8,192 B | 16 B | `[-8, 10, -15, 3]` |
| TFLM | CMSIS-NN FullyConnected int8 | 5,499 | 5,499 | 63,928 B | 10,656 B | 1,024 B reserved / 580 B reported used | `[-8, 10, -16, 4]` |

On this small MLP, CMSIS-NN is **35.4% fewer cycles than BakeNN portable C**
(1.55x faster).  Its price is roughly 3.1x linked Flash and 30.1% more
linked SRAM.  The raw output differs at two of four bytes by one output LSB;
this is the already-recorded BakeNN-versus-TFLite integer-rounding difference,
not a CMSIS-NN-only mismatch.  TFLM reference and CMSIS-NN outputs had the
same FNV-1a checksum on this model.

For a direct kernel-library comparison on the same node, the reference TFLM
image measured 9,431 median cycles, 63,188 B Flash, 10,624 B linked SRAM, and
the same `0x89a44813` output checksum.  CMSIS-NN therefore reduced TFLM's
measured MLP invoke time by 41.7% while adding 740 B to this linked image.

## Reproduction

- Board: `nrf52840dk-1.saclay.iot-lab.info`, Nordic nRF52840, 64 MHz Cortex-M4F
- TFLM: Zephyr 2.7.5 module, commit `9156d050927012da87079064db59d07f03b8baf6`
- CMSIS-NN: v4.0.0, commit `ca5dc34313be2ee5c46652917c30baac96c52621`
- Toolchain: Zephyr SDK 0.13.1, `arm-zephyr-eabi-gcc` 10.3.0, `-O2`, section GC
- Model SHA-256: `dd18dfbac4f2abc237aa2eb70809193e7d1dfa0d85c7d11b8eb3b499810df592`
- Input SHA-256: `af9613760f72635fbdb44a5a0a63c39f12af30f950a6ee5c971be188e89c4051`
- Warmup/measured invokes: 8 / 101

The machine-readable result is
[`iotlab_447626_cmsis_nn.json`](iotlab_447626_cmsis_nn.json), and its schema
passes `validate_result.py`.  Stack and whole-firmware peak SRAM remain
explicitly unmeasured; the reported linked SRAM and arena values are not a
claim of full application memory fit.
