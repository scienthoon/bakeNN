# BakeNN target-build smoke

These fixtures verify target compilation without claiming physical-target
performance.  Generate and link the deterministic model for ARM or RISC-V:

```bash
PYTHONPATH=src python examples/targets/generate_smoke.py \
  --target cortex-m0plus --output build/m0 --cross-build
PYTHONPATH=src python examples/targets/generate_smoke.py \
  --target cortex-m4 --output build/m4 --cross-build
PYTHONPATH=src python examples/targets/generate_smoke.py \
  --target rv32imc --output build/rv32 --cross-build
```

The `cross` directory contains the freestanding ELF, linker map, and a JSON
report.  The report includes linked Flash-load and static-SRAM bytes plus the
undefined/heap/float-symbol audit.  Its cycle and stack fields remain explicitly
unmeasured.

Generate an ESP-IDF project without a board:

```bash
PYTHONPATH=src python examples/targets/generate_smoke.py \
  --target esp32s3 --output build/esp32s3 --esp-nn --esp-idf
cd build/esp32s3/esp_idf
idf.py set-target esp32s3
idf.py build
idf.py size-components
```

The ESP runner calls the model once and, when flashed later, prints the cycle
counter, FreeRTOS stack high-water mark, arena size, and output bytes.  An
ESP-IDF build proves source/toolchain/link compatibility only; it does not prove
latency, energy, cache behaviour, or physical peak stack usage.

`--esp-nn` enables the pinned, direct ESP-NN source backend. On ESP32-S3 the
smoke graph selects its Conv2D and DepthwiseConv2D SIMD implementations; on the
original ESP32 it selects Espressif's optimized generic Conv/Depthwise C. The
same command for ESP32-C3 proves deterministic BakeNN fallback because the
pinned ESP-NN release does not implement that target. The generated component
contains the required source closure, headers and license, so neither TFLM nor
an online component-manager fetch is required.

Generate an IoT-LAB nRF52840DK Zephyr benchmark project:

```bash
PYTHONPATH=src python examples/targets/generate_smoke.py \
  --target cortex-m4 --output build/iotlab --zephyr-board nrf52840dk_nrf52840
cd build/iotlab/zephyr
west build -p always -b nrf52840dk_nrf52840 .
```

The runner initializes the input to its quantized zero point, performs eight
warmups and 101 measured inferences, then prints first/median/p95 cycles, arena
bytes, current-thread unused stack bytes, and the exact INT8 output over UART.
For FIT IoT-LAB, reserve one `nrf52840dk` node at Saclay and flash
`build/zephyr/zephyr.elf`. The reservation clock starts immediately, so build
the firmware before submitting an as-soon-as-possible experiment.
The fixture uses `--kernel-policy auto` by default so that the Cortex-M4 DSP
kernel is actually exercised. Pass `--kernel-policy portable` to produce the
same model with the baseline kernel for an A/B measurement.

For an FP32 PyTorch model whose `Linear` layers should use the pinned direct
CMSIS-NN backend, opt in at compilation time:

```python
options = bakenn.CBackendOptions(
    kernel_policy=bakenn.KernelPolicy.AUTO,
    enable_cmsis_nn=True,
    target=bakenn.CORTEX_M4,
)
compiled = bakenn.compile_torch_ptq(
    model,
    example_input,
    calibration_data,
    "build/model",
    backend_options=options,
    target=bakenn.CORTEX_M4,
)
project = bakenn.export_zephyr_project(
    compiled.artifacts,
    bakenn.CORTEX_M4,
    "build/zephyr",
)
```

BakeNN copies the exact CMSIS-NN FC source closure, CMSIS Core headers, and
Apache-2.0 license files into the generated artifact. The firmware links those
sources directly; TFLM is not required. `AUTO` falls back to BakeNN kernels if
the target, tensor shape, or quantization contract is incompatible. The FC v4
path requires one shared weight/requantization scale per `Linear`; the opt-in
therefore selects per-tensor Linear weight PTQ instead of BakeNN's default
per-output-channel policy. Use an explicit `PTQOptions` value when accuracy
policy must be selected independently of the backend.

When integrating the generated directory manually with CMake, consume all
three variables from `bakenn_sources.cmake`: `BAKENN_MODEL_SOURCES`,
`BAKENN_MODEL_INCLUDE_DIRS`, and `BAKENN_MODEL_COMPILE_DEFINITIONS`. The last
one is required to keep CMSIS-NN's fixed-width DSP loads inlined on
freestanding/minimal-libc builds.
