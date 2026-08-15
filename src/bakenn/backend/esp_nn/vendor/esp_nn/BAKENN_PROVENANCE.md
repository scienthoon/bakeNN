# Vendored ESP-NN provenance

- Upstream: https://github.com/espressif/esp-nn
- Release version: 1.2.6
- Git revision: `c0876179f1cf4b4b9073b4f81cb65c8051ccb476`
- License: Apache-2.0; see `LICENSE` in this directory

BakeNN vendors the source and public headers so generated ESP-IDF projects are
self-contained and reproducible. The source is not modified. BakeNN's adapter,
capability checks, static scratch planning and selection metadata live outside
this directory.

`CONFIG_NN_OPTIMIZED=1` selects the upstream optimized implementation.
`CONFIG_NN_SKIP_NUDGE` must remain disabled because BakeNN's
`bakenn.int8.v1` contract requires TFLM-compatible double rounding.
