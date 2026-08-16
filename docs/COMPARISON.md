# BakeNN compared with TFLM and Edge Impulse EON

## Bottom line

BakeNN is the stronger deployment tool when one trained PyTorch model is fixed
at firmware-build time, shapes are static, and the product values small,
inspectable C firmware over runtime model interchange. TFLM is stronger when a
single firmware image must load different `.tflite` files or needs its broader
operator and vendor ecosystem. EON is stronger when a team wants Edge
Impulse's complete data-to-device platform; BakeNN is stronger when compilation
must remain local, open and independent of TFLite and a hosted service.

This is not a comparison between “working software” and a straw-man baseline.
TFLM and EON are mature systems. BakeNN wins a narrower problem by accepting a
narrower contract and compiling it more completely.

## Architecture

| Property | BakeNN | TFLM / LiteRT Micro | Edge Impulse EON |
|---|---|---|---|
| Primary input | FP32 PyTorch `eval()` model plus representative data | Fully quantized LiteRT/TFLite FlatBuffer | LiteRT/TFLite FlatBuffer inside the Edge Impulse deployment flow |
| Target artifact | Standalone C11 model, weights and selected kernels | FlatBuffer bytes plus C++17 interpreter/runtime and registered operators | Generated C++ plus Edge Impulse SDK integration |
| Graph decisions | Host compile time | Runtime initialization and interpreter execution; optional offline memory metadata exists | Edge Impulse compilation service |
| Memory plan | Static offsets, lifetimes, scratch and budget gate emitted by BakeNN | Caller-provided arena managed by TFLM; recording and offline-allocation facilities are available | Compiled plan; implementation is not fully open |
| Kernel path | Portable C or direct pinned CMSIS-NN/ESP-NN calls | Reference or target-optimized TFLM kernels | TFLM kernels/optimizations under the hood |
| Model update | Rebuild firmware | Replace model data when ABI/operator support remains compatible | Rebuild deployment library |
| Toolchain on device | C11 | C++17 | C++ |
| Compiler availability | Apache-2.0, local source | Apache-2.0, local source | Hosted proprietary compiler; generated deployment code is provided to the user |

Google's current LiteRT Micro overview describes a C++17 runtime for 32-bit
platforms and a workflow where a converted model is embedded as a C byte array
and interpreted by the device library. A C array is therefore packaging for the
FlatBuffer, not model-specialized inference source. See the
[LiteRT Micro overview](https://developers.google.com/edge/litert/microcontrollers/overview)
and [model conversion guide](https://developers.google.com/edge/litert/microcontrollers/build_convert).

Edge Impulse documents EON's input as a LiteRT FlatBuffer and its output as
`.cpp`/`.h` containing unpacked weights and prepare/invoke functions. Its launch
article also states that EON continued to use TFLM kernels and optimizations,
which is why the example's latency stayed the same while memory fell. See the
[current EON documentation](https://docs.edgeimpulse.com/studio/projects/deployment/eon-compiler)
and [EON launch article](https://edgeimpulse.com/blog/introducing-eon/).

## What BakeNN removes

For its supported static contract, BakeNN resolves the following before target
execution:

- operator order and exact tensor shapes;
- quantization scales, zero-points and Q31 multipliers/shifts;
- activation lifetimes, arena offsets, aliasing and backend scratch;
- kernel implementation and packed-constant layout;
- fused activation bounds and compile-time accumulator safety proofs;
- the exact C call graph and selected source closure.

The device receives direct calls with constant dimensions and addresses. There
is no FlatBuffer parser, operator resolver, runtime tensor planner or target
float fallback. This is more than removing one dispatch loop: it changes the
deployment unit from “runtime plus model data” to “the compiled model.”

## Measured evidence

Only rows explicitly marked **physical** are performance evidence. A host test
proves numerical correctness. A cross-build proves target toolchain
compatibility. Neither is an MCU latency result.

### Physical nRF52840, frozen 32→16→4 INT8 FC

Same board, 64 MHz clock, model constants, qparams, input bytes, output bytes,
CMSIS-NN FC family, 8 warmups and 101 measured calls:

| Path | Median cycles | Linked Flash | Linked static SRAM | Output |
|---|---:|---:|---:|---|
| BakeNN direct CMSIS-NN | **3,786** | **20,920 B** | **8,540 B** | byte-exact |
| TFLM + CMSIS-NN | 5,418 | 69,640 B | 11,040 B | byte-exact |

For this exact image BakeNN used 30.1% fewer cycles, 70.0% less linked Flash
and 22.6% less linked static SRAM. The cycle delta is the complete generated
call path versus the complete TFLM invoke path around the matching kernel
family; it must not be presented as a universal per-operator dispatch tax.
Raw UART, model/input hashes and limitations are in the
[nRF52840 report](../benchmarks/tflm_compare/results/iotlab_447626_direct_cmsis_fc.md).

### Physical original ESP32, trained MobileNetV2-0.25

Same frozen quantized graph, real-zero input, output bytes, 240 MHz clock, 8
warmups and 101 measured calls:

| Path | Median latency | App binary | Linked DRAM | Output |
|---|---:|---:|---:|---|
| BakeNN portable C | 285.546 ms | 459,088 B | 31,900 B | byte-exact |
| BakeNN + ESP-NN | **97.685 ms** | **465,296 B** | **31,900 B** | byte-exact |
| TFLM + ESP-NN | 98.891 ms | 665,504 B | 95,332 B | byte-exact |

Direct BakeNN-to-ESP-NN was 1.22% lower latency than TFLM+ESP-NN for this
artifact, while reducing the app binary by 30.1% and linked DRAM by 66.5%.
The small latency gap on the larger convolutional model is important: when
optimized kernels dominate total work, AOT's most reliable advantage is often
Flash/RAM and integration simplicity rather than a dramatic speedup. See the
[full ESP32 report](../benchmarks/esp32/results/mobilenet_v2_025_cifar10_esp32_tflm_espnn.md).

### Boardless Cortex-M4: trained MNIST versus microTVM AOT+USMP

This additional baseline uses Apache TVM 0.16.0 rather than TFLM. The same
trained FP32 checkpoint, 160-image calibration corpus, quantized graph, input
bytes, TFLite operator semantics and CMSIS-NN 4.0.0 source family were frozen
for both paths. microTVM used the AOT C interface and USMP
`greedy_by_size`; BakeNN generated direct calls from its execution plan.

| Path | Linked Flash | Linked static SRAM | Planned workspace | Host output differential |
|---|---:|---:|---:|---:|
| BakeNN direct CMSIS-NN | **12,088 B** | 4,864 B | 4,064 B | 0 / 1,000 bytes |
| microTVM AOT+USMP+CMSIS-NN | 17,456 B | **4,862 B** | 4,064 B | 0 / 1,000 bytes |

For this exact Cortex-M4 ELF, BakeNN linked 30.8% less Flash while the static
SRAM totals were effectively equal. This is compiler/linker evidence, not a
speed claim: no Cortex-M4 cycles were measured. The result is nevertheless a
full-model proof, not an operator toy—two convolutions, two max pools, reshape
and fully connected were lowered, and microTVM's emitted C called
`arm_depthwise_conv_wrapper_s8`, `arm_convolve_wrapper_s8`,
`arm_max_pool_s8` and `arm_fully_connected_s8`. See the
[protocol, generated sources and hashes](../benchmarks/microtvm_compare/README.md).

## Development and audit differences observed in this repository

The checked-in comparisons exposed concrete integration work on the TFLM side:

- the selective `MicroMutableOpResolver` capacity and registrations had to be
  kept synchronized with the model;
- the FlatBuffer operator versions had to match the TFLM snapshot shipped by
  the selected RTOS/SDK;
- the TFLM snapshot, CMSIS-NN headers, wrapper status names and registration
  symbols needed compatibility glue;
- the caller had to reserve a tensor arena and validate allocation on target.

The first resolver error was our configuration mistake, not a TFLM bug. It is
still a real workflow distinction: BakeNN derives required kernels from its
verified graph and emits only that closure. There is no resolver capacity or
operator-version matrix in the generated firmware.

Audit scope differs in the same way. A TFLM review spans the FlatBuffer,
interpreter, resolver, memory planner, selected kernels and application
configuration. A BakeNN review spans the generated C/weights, manifest and the
compiler version that produced them. BakeNN does not make safety certification
automatic, but its fixed C call graph and explicit memory report are materially
easier inputs to static analysis and reproducible review.

## Memory-budget distinction

TFLM does have memory measurement and planning tools, including its greedy
arena planner and offline allocation metadata. The BakeNN distinction is not
that TFLM “cannot know memory.” It is that BakeNN makes model-level constants,
activation peak, backend scratch and alignment first-class compiler outputs and
can reject a declared budget before firmware integration. Final whole-firmware
Flash, unrelated globals and peak stack still require the target ELF/map and,
for stack, target measurement.

## Where TFLM wins

- Much broader operator coverage and years of deployment experience.
- Standard `.tflite` interchange and model-only update workflows.
- Mature integrations across many vendor SDKs and optimized kernel libraries.
- A better fit when one firmware runtime must accept multiple models.

## Where EON wins

- A complete commercial platform: data collection, labeling, DSP feature
  extraction, training, optimization, fleet/deployment workflow and support.
- Broader model/board coverage inherited from Edge Impulse and TFLM.
- Mature product experience and vendor validation that BakeNN does not yet
  have.

## Where BakeNN wins

- Direct `torch.export` capture from an existing PyTorch workflow without an
  ONNX/TFLite round trip.
- Fully local, Apache-2.0 compilation with no model upload or hosted compiler.
- Standalone C11 instead of a FlatBuffer runtime or generated C++ SDK bundle.
- Explicit static-memory contracts, budget failure and readable deployment
  artifacts.
- Direct CMSIS-NN/ESP-NN source closure without retaining TFLM around the
  selected kernels.
- Reproducible integer-reference ↔ generated-C byte-exact testing for the
  supported arithmetic profiles.

## Deliberate tradeoff

BakeNN is static batch one, fixed shape, and one public input/output. It rejects
unsupported semantics instead of falling back. Those constraints are the
source of its strongest properties: fixed offsets, fixed call graph, no model
loader, compile-time memory proofs and model-specialized C. They are the wrong
tradeoff for dynamic models and the right tradeoff for firmware that already
rebuilds whenever its linked model changes.

## Evidence index

- [Physical measurements](../benchmarks/physical/README.md)
- [Boardless cross-build evidence](../benchmarks/cross_build/README.md)
- [microTVM AOT+USMP comparison](../benchmarks/microtvm_compare/README.md)
- [Benchmark protocol](../benchmarks/tflm_compare/README.md)
- [Reproduction guide](../REPRODUCING.md)
