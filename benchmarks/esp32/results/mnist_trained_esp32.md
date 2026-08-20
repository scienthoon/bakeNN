# Trained MNIST full model on a physical ESP32

The frozen trained MNIST checkpoint was regenerated at submission commit
`23f3a4f744135ad02bc738a77aee531f7ff2a751`, built with ESP-IDF v5.5.4,
flashed to a physical original ESP32, and run over all 100 frozen physical-test
inputs. The board produced exactly the same 1,000 INT8 output bytes as the
BakeNN Python reference.

The machine-readable record is
[`mnist_trained_esp32.json`](mnist_trained_esp32.json), and the reset-to-exit
UART transcript is
[`mnist_trained_esp32_uart.txt`](mnist_trained_esp32_uart.txt).

## Result

| Metric | Physical ESP32 result |
|---|---:|
| Samples | 100 |
| Correct | 99 |
| Accuracy | 99.00% |
| Compared output bytes | 1,000 |
| Mismatched output bytes | **0** |
| Output FNV-1a | `0x55fb9e60` |
| First inference | 3,311,839 cycles / 20.698994 ms |
| Median inference | 3,165,617 cycles / 19.785106 ms |
| p95 inference | 3,165,617 cycles / 19.785106 ms |
| Median throughput | 50.543 inference/s |
| Stack high-water mark | 6,752 free bytes |

Latency used the first frozen corpus sample. The runner performed one cold
call, eight warmups, and 101 measured calls, sorting the measured cycle counts
for median and p95. The subsequent correctness pass ran all 100 inputs and
compared every one of the ten output bytes per input. `vTaskDelay(1)` is outside
each measured interval.

## Board and toolchain

| Property | Value |
|---|---|
| Board product label | PSRAM Type-C ESP32 CAM |
| USB bridge | QinHeng CH340, VID:PID `1a86:7523` |
| SoC | ESP32-D0WDQ6 revision v1.1, dual-core Xtensa LX6 |
| Configured CPU clock | 160 MHz |
| Crystal | 40 MHz |
| Physical Flash | 4 MB |
| Firmware header Flash size | 2 MB |
| ESP-IDF | v5.5.4, commit `735507283d5b2f9fb363a1901172dbd9e847945d` |
| Compiler | xtensa-esp-elf GCC 14.2.0, crosstool-NG `esp-14.2.0_20260121` |
| Optimization | ESP-IDF size profile (`-Os`); BakeNN model component final `-O2` |
| Flash interface | DIO at 40 MHz |
| Capture | 115,200 baud, started 2026-08-20 21:27:37 KST |

The chip and Flash values were queried with esptool as part of this run. PSRAM
was not enabled or used.

## Flash and SRAM

| Metric | Bytes |
|---|---:|
| Application binary | 235,968 |
| ELF total image | 235,859 |
| Flash data | 118,968 |
| Flash code | 60,824 |
| IRAM | 46,247 |
| DRAM | 17,148 |
| Static SRAM, IRAM + DRAM | 63,395 |
| RTC slow memory | 56 |
| BakeNN model component Flash | 8,871 |
| Embedded inputs + labels + expected outputs | 79,500 |
| BakeNN activation arena | 3,920 |
| BakeNN constants | 4,508 |

The app binary includes the 79,500-byte on-device validation payload; it is not
model storage. The model component's 8,871 bytes comprise 4,508 bytes of Flash
data and 4,363 bytes of Flash code.

The generated UART key is named `stack_high_water_words` for historical
reasons. In ESP-IDF v5.5.4, `uxTaskGetStackHighWaterMark(NULL)` returns the
minimum free stack in **bytes**, unlike the standard FreeRTOS word-based API.
Accordingly, the measured value is recorded as 6,752 free bytes. The configured
main-task stack is 8,192 bytes.

## Frozen provenance

| Artifact | SHA-256 |
|---|---|
| Submission commit | `23f3a4f744135ad02bc738a77aee531f7ff2a751` |
| Checkpoint file | `a2f08c02a718d2e6cbe590ad442891db04308a3c82a2c791b8e0fc02186ea498` |
| Logical checkpoint tensors | `733130916e926da785afc63da0786d7613b0db808c3482398ae84b41ab706840` |
| Calibration corpus | `06ce2a1aaaa0f75e566b9494c17b3c1fce4063272d1ce4e879c24b868bbb4e29` |
| Quantized physical input | `27c96cc7e1e5b49058b8183781af349c1773b89cbfa4791b6eb098397111f653` |
| Physical labels | `b51f622729467a10be28516c6d6a8fc3ec9f6ab8be247e2b9866312af8998595` |
| Expected/board output | `cdbbfc38bf5f17a2867b17a8a28e643043e50733063e246b0bc85cbeaae30ba4` |
| ESP32 generated component set | `93cd905ef56862a9ba628b24e8355e9e2481d178cb6107384e282ab08dc53fcf` |
| Benchmark contract | `45d36e5549f1a46e5b830daeea786a2a1d10b80d370ad4474c5f8a3a1baad6d9` |
| Firmware ELF | `a8c68d7b5c876819b39fb2ea6bd686680430c2787b23e4da281eebc5cb16c1ef` |
| Application binary | `fc43434ee8784e557695c66ae02ab06b8e1703454b85fa9dc02e3ed4376291d2` |
| Link map | `2af25f9ef37c44779c137057cbf5e94b2671747b80cd57afad93b7fea78f053a` |
| UART transcript | `8965d215c8148b3287fd33e92b96672170b42b8dce70dc637240418e71cc948d` |

The ESP32 component-set hash covers the 19 files actually supplied to the
ESP-IDF model component, including generated model sources, manifest, component
CMake file, and pinned ESP-NN source subset. It uses the
`bakenn.mnist.generated-artifact-set.v1` domain implemented by
`examples/mnist/evidence_utils.py`.

## Reproduction

From a clean checkout of the submission commit, install the Torch example
dependencies and regenerate the project:

```bash
git checkout 23f3a4f744135ad02bc738a77aee531f7ff2a751
python -m pip install -e '.[torch]' 'torchvision==0.25.0'
PYTHONPATH=src python examples/mnist/generate_esp32_benchmark.py
```

The generator validates the checkpoint's logical tensor hash, calibration
corpus hash, quantized input bytes, expected Python output bytes, and expected
FNV-1a before producing the project.

Build, inspect memory, flash, and monitor with ESP-IDF v5.5.4:

```bash
export IDF_PATH=/path/to/esp-idf-v5.5.4
. "$IDF_PATH/export.sh"
cd build/mnist_esp32_physical/esp_idf
idf.py set-target esp32
idf.py build
idf.py size
idf.py size-components
idf.py -p "$PORT" flash monitor
```

The acceptance lines are:

```text
BAKENN_MNIST target=esp32 cpu_mhz=160 samples=100 correct=99 accuracy_bp=9900 compared_bytes=1000 mismatches=0 first_cycles=3311839 median_cycles=3165617 p95_cycles=3165617 stack_high_water_words=6752 arena=3920
BAKENN_MNIST_OUTPUT_FNV1A=0x55fb9e60
BAKENN_MNIST_PROVENANCE checkpoint=733130916e926da785afc63da0786d7613b0db808c3482398ae84b41ab706840 calibration=06ce2a1aaaa0f75e566b9494c17b3c1fce4063272d1ce4e879c24b868bbb4e29 input=27c96cc7e1e5b49058b8183781af349c1773b89cbfa4791b6eb098397111f653 expected=cdbbfc38bf5f17a2867b17a8a28e643043e50733063e246b0bc85cbeaae30ba4
```

This is a result for the exact board, clock, firmware, generated component,
toolchain, and corpus above. Host execution time was not used as MCU
performance evidence.
