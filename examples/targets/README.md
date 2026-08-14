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
  --target esp32s3 --output build/esp32s3 --esp-idf
cd build/esp32s3/esp_idf
idf.py set-target esp32s3
idf.py build
idf.py size-components
```

The ESP runner calls the model once and, when flashed later, prints the cycle
counter, FreeRTOS stack high-water mark, arena size, and output bytes.  An
ESP-IDF build proves source/toolchain/link compatibility only; it does not prove
latency, energy, cache behaviour, or physical peak stack usage.
