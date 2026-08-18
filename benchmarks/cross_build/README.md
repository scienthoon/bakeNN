# Boardless cross-build evidence

This directory records target compilation and link compatibility only. A
cross-build can prove that generated code, vendor sources, headers, linker
configuration and ABI form a valid target ELF. It cannot prove physical
latency, cache behavior, energy, clock configuration or stack high-water mark.

Current CI cross-builds:

- host GCC and Clang strict C11 with ASan/UBSan differential tests;
- `arm-none-eabi-gcc` for Cortex-M0+ and Cortex-M4;
- GNU RISC-V for RV32IMC;
- ESP-IDF projects for ESP32, ESP32-S3 and ESP32-C3.

The trained MNIST original-ESP32 full-model build is recorded in
[`results/mnist_esp32_cross_build.json`](results/mnist_esp32_cross_build.json).
Its cycle and physical-output fields are explicitly unmeasured. Do not place
this result in a performance table.

The same trained MNIST graph was also built for Cortex-M4 through BakeNN and
through Apache TVM 0.16.0 AOT+USMP+CMSIS-NN. The generated microTVM C matched
the BakeNN integer reference over all 1,000 output bytes and both paths formed
freestanding ELFs. See the
[microTVM comparison](../microtvm_compare/README.md). Its linked section sizes
are valid cross-build evidence; its cycles remain unmeasured.
