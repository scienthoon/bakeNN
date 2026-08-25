# Third-party notices

BakeNN itself is licensed under Apache-2.0.  Optional generated target bundles
contain pinned, redistributable subsets of the following upstream projects.

## CMSIS-NN

- Project: Arm CMSIS-NN
- Upstream: <https://github.com/ARM-software/CMSIS-NN>
- Version: 4.0.0
- Revision: `ca5dc34313be2ee5c46652917c30baac96c52621`
- License: Apache-2.0
- Packaged license:
  `src/bakenn/backend/cmsis_nn/vendor/cmsis_nn/LICENSE.txt`

BakeNN packages only the source closure needed by selected FullyConnected,
Conv2D, DepthwiseConv2D, AveragePool2D and MaxPool2D kernels.
Four copied CMSIS-NN files contain documented freestanding/compiler
compatibility changes and one BakeNN support source is added. Exact upstream
and patched hashes, purposes, and modification notices are recorded in
`src/bakenn/backend/cmsis_nn/vendor/cmsis_nn/BAKENN_PROVENANCE.md`.

## CMSIS-Core

- Project: Arm CMSIS_5 / CMSIS-Core headers
- Upstream: <https://github.com/ARM-software/CMSIS_5>
- Version: 5.9.0
- License: Apache-2.0
- Packaged license:
  `src/bakenn/backend/cmsis_nn/vendor/cmsis_core/LICENSE.txt`

The headers provide the compiler and Arm intrinsic definitions required by the
vendored CMSIS-NN source subset.

## ESP-NN

- Project: Espressif ESP-NN
- Upstream: <https://github.com/espressif/esp-nn>
- Version: 1.2.6
- Revision: `c0876179f1cf4b4b9073b4f81cb65c8051ccb476`
- License: Apache-2.0
- Packaged license: `src/bakenn/backend/esp_nn/vendor/esp_nn/LICENSE`
- Provenance record:
  `src/bakenn/backend/esp_nn/vendor/esp_nn/BAKENN_PROVENANCE.md`

The vendored upstream source is unmodified.  BakeNN's adapters, capability
checks, scratch planner and code generator are maintained outside the vendor
directory.

## Development and benchmark dependencies

PyTorch, torchvision, TensorFlow Lite/LiteRT, FlatBuffers, Zephyr and ESP-IDF
are not copied into generated target artifacts. They are optional host
frontend, test, comparison or target-build dependencies and retain their own
licenses. The strict TFLite importer uses the optional host schema and
FlatBuffers packages, while LiteRT is used only as an external differential
oracle. Generated portable C has no TFLite, FlatBuffers or interpreter
dependency.

Generated target artifacts copy the applicable third-party license next to the
selected source closure.  The wheel and source distribution also include the
packaged license and provenance files so downstream users can audit them before
redistribution.
