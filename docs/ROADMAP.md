# BakeNN roadmap

Status date: 2026-08-14

This document starts from the code that exists today and defines the work
required before BakeNN can make target-performance or broad compatibility
claims. It is an execution plan, not a promise that unmeasured kernels are
faster.

## Current baseline

The following are implemented and covered by the current host test suite:

- static batch-one INT8 graph, verification, liveness, arena and scratch plan;
- PyTorch FP32 eval capture and deterministic min/max PTQ;
- per-tensor affine INT8 activations, per-channel symmetric INT8 weights,
  INT32 bias/accumulator proofs and `bakenn.int8.v1` Q31 requantization;
- standalone heap-free C11 generation and an independent Python integer
  reference;
- Conv2D including groups, depthwise Conv2D, Conv1D including groups, Linear,
  2D/1D pooling, static broadcast Add/Mul, Clamp, Requantize, Sigmoid,
  HardSigmoid, HardSwish, SiLU, Pad2D, ReduceMean, shape/view operations,
  static Slice/Crop, Concatenate, nearest/Q15-bilinear Resize2D, grouped
  ConvTranspose2D and
  Softmax;
- generic optimized Linear/1x1/depthwise kernels;
- Cortex-M4 DSP Linear, 1x1 Conv, depthwise 3x3, im2col 3x3 Conv, global
  average-pool and 2x2 max-pool candidates;
- ARM/RISC-V cross-link and symbol audit, plus ESP-IDF project generation;
- Python reference versus generated-C byte-exact tests.

The current full suite passes 233 tests and 6 subtests. Cortex-M4 ELF
disassembly contains the expected `smlad` instructions. No physical target
cycle, stack, energy or BakeNN-versus-TFLM result has been measured.

## Rules that apply to every phase

1. Portable C remains the correctness baseline.
2. A target implementation must preserve the declared arithmetic-profile ID
   or introduce a new profile; approximate rounding may not silently claim
   `bakenn.int8.v1`.
3. Unsupported semantics fail during host compilation. There is no target-side
   float or interpreter fallback.
4. Python reference, portable C and every selected target kernel must be
   byte-exact for accepted inputs.
5. Host latency, instruction counts and vendor benchmark numbers are not
   physical BakeNN measurements.
6. A measured cost entry must record target, workload, toolchain, flags,
   firmware revision, raw evidence and output mismatch count.
7. Existing tests may not be weakened to accept a new kernel.

## R1 — Make kernel selection honest

Priority: immediate, no physical board required.

The selector currently has three policies: `PORTABLE`, `AUTO` and
`REQUIRE_OPTIMIZED`. `AUTO` filters capability predicates and chooses the
highest static priority. `TargetDescriptor.measured_costs` is recorded but is
not yet consulted.

Work:

- separate selection basis from failure policy, for example:
  - `PORTABLE`;
  - `STATIC_PRIORITY` as explicit experimental opt-in;
  - `MEASURED` for cost-table selection;
  - `REQUIRE_OPTIMIZED` as a coverage/audit mode;
- define a versioned canonical workload key containing op kind, relevant
  shapes, stride, dilation, groups, padding, qparam profile and memory class;
- record `selection_basis` and the exact matched cost entry in the manifest;
- make measured selection optimize one declared objective, initially latency,
  while enforcing Flash and SRAM budgets;
- when no matching measured entry exists, choose portable rather than guessing;
- if a selected candidate exceeds a resource budget, retry the next feasible
  candidate instead of failing after selection;
- add deterministic tests for exact matches, absent measurements, stale
  toolchain/flags, resource rejection and tie handling.

Acceptance:

- no mode named or documented as measured/fastest selects from static priority;
- an empty measured table deterministically produces portable selections;
- every non-portable measured selection cites immutable evidence in the
  generated manifest.

## R1A — Model-specialized portable Conv

Priority: immediate after the R1 selector contract. The compiler, generated-C
shape and host differential tests can be completed without a physical board;
promotion into measured `AUTO` selection still requires R2/R3 evidence.

Motivation:

- the generic portable Conv is intentionally broad, but it pays for that
  breadth inside the hottest loop through coordinate checks, flattened-index
  arithmetic and general stride/dilation/group handling;
- legacy PoTNN showed that baking a fixed model's dimensions and constants into
  generated code can expose much more work to the C compiler;
- PoTNN's shift-only arithmetic is not an INT8 replacement. BakeNN may reuse
  its static-specialization and unrolling ideas, but it must preserve
  `bakenn.int8.v1`, per-channel requantization and byte-exact output;
- the previously reported host `13x` gap is a diagnostic lead, not performance
  evidence: it compared a generic INT8 kernel with a fully unrolled PoT kernel
  under different arithmetic and code-generation conditions.

### OPT-01 — Trustworthy local benchmark

- run portable, each AOT candidate and the legacy comparison with the same
  input corpus, compiler, optimization flags, LTO policy and translation-unit
  boundaries;
- consume every output through a checksum or externally visible sink and
  verify expected bytes so dead-code elimination cannot erase either model;
- record the exact selection policy and selected kernel ID;
- report generated C size, final object/ELF `.text` and `.rodata`, rather than
  comparing source length;
- label host timing as a smoke/regression signal only. It may reject an obvious
  regression but may not populate an MCU measured-cost table;
- compare legacy PoTNN separately from TFLM and BakeNN because its weight
  representation and inference arithmetic implement a different numeric
  contract.

### OPT-02 — Clean up the generic portable Conv

Keep one fully general fallback, but remove avoidable work without changing its
accepted semantics:

- hoist group sizes, invariant strides and base offsets out of inner loops;
- use pointer increments or precomputed row bases instead of recomputing full
  flattened indices for every MAC;
- when a kernel coordinate is outside the input, skip the whole input-channel
  loop: padding is `input_zero_point`, so its centered contribution is exactly
  zero;
- split interior pixels from border pixels when that reduces repeated bounds
  checks without duplicating the numeric implementation;
- use `int32_t` coordinates only where host-side range proofs establish that
  every intermediate is safe; retain a safe generic path or reject impossible
  plans rather than relying on signed overflow;
- preserve grouped Conv, stride, dilation and asymmetric-padding coverage in
  the fallback.

### OPT-03 — Static direct 3x3 Conv candidate

Add a separately versioned, narrow candidate rather than turning the portable
kernel into a maze of special cases.

Initial capability:

- NHWC input/output and OHWI INT8 weights;
- groups 1, kernel 3x3, dilation 1, stride 1 and compile-time padding;
- output-channel blocking by two, with a tested odd-channel tail;
- no scratch allocation in the first version;
- per-channel multiplier/shift, asymmetric input zero point and fused clamp.

Proposed IDs:

```text
optimized.conv2d_3x3_direct_o2.v1
conv2d_3x3_ohwi_o2_interleaved_v1
```

The generated body or wrapper must expose dimensions, strides and padding as
compile-time constants. Interior and border loops may be separate. Output-pixel
blocking, stride 2 and scratch/im2col variants are later immutable IDs, not
silent changes to v1.

### OPT-04 — Budgeted partial and full unrolling

Treat unrolling as a compile-time candidate selected under a Flash budget, not
as a universal optimization.

- first specialize loop bounds and constant addresses while retaining compact
  loops;
- then add partial unrolling for kernel positions/output-channel blocks;
- permit full unrolling only for small, fixed layers whose estimated and final
  code size fit the declared budget;
- remove zero-weight MACs only when the resulting integer expression remains
  exactly equivalent and accumulator proofs are recomputed;
- bake INT8 weights and offsets as constants, but keep ordinary INT8 multiply;
  do not relabel PoT shift arithmetic as INT8;
- reject or fall back when source size, compiler time, Flash growth or branch
  fan-out exceeds deterministic limits.

Proposed IDs:

```text
aot.conv2d_3x3_partial_unroll.v1
aot.conv2d_3x3_full_unroll.v1
```

### OPT-05 — Validation and promotion

Every candidate must pass:

- hand goldens and randomized Python-reference/portable/candidate byte
  differential tests, including INT8 extremes, nonzero zero points, border
  padding, odd channels and positive/negative requantization shifts;
- strict C11 under GCC and Clang at `-O0`, `-O2` and `-Os`, plus ASan/UBSan
  where executable;
- cross-link, undefined-symbol and disassembly inspection for each supported
  cross toolchain;
- deterministic source/constant generation and explicit Flash/scratch limits;
- object/ELF size reporting for both the winning candidate and portable
  fallback.

Completion has two levels:

1. **Host-complete:** candidates are byte-exact and available through explicit
   experimental/static selection.
2. **Measured-default:** an exact target/workload/toolchain cost entry proves a
   candidate wins within Flash/SRAM budgets; only then may measured `AUTO`
   select it.

## R2 — Hardware benchmark package and remote result intake

Priority: immediate. It can be implemented without owning a board; completion
of the measurements requires access to physical hardware somewhere.

Work common to every target:

- generate a fixed input corpus plus SHA-256 identity;
- run portable and target kernels in the same firmware image where practical;
- perform warm-up, repeated samples, median and p95 collection;
- capture output bytes and require zero mismatch;
- measure final ELF Flash, static SRAM, arena, scratch and stack watermark
  without double-counting;
- emit a versioned JSON result with compiler, flags, linker script, clock,
  memory placement, wait states and firmware revision;
- validate and import a result without trusting unproven derived fields;
- retain raw UART logs, ELF, map and disassembly as evidence artifacts.

Cortex-M work:

- add a small DWT `CYCCNT` benchmark runner;
- provide board adapters for clock, UART, startup and linker script rather than
  pretending one Cortex-M4 profile describes every MCU;
- begin with a common STM32F4-class board adapter, but keep the measurement
  keyed to the exact chip and memory placement.

ESP work:

- finish the existing ESP-IDF runner contract for `esp32s3`;
- record `esp_timer`/cycle-counter source, FreeRTOS stack high-water mark and
  internal-RAM versus PSRAM placement;
- make board output directly consumable by the common result validator.

Remote workflow:

```text
BakeNN creates immutable benchmark bundle
        -> board owner or hardware CI flashes it
        -> UART/result artifacts are returned
        -> validator checks identity, provenance and output
        -> accepted measurements enter a separate reviewed cost-data file
```

Acceptance:

- a person with the declared board can build, flash and return a result without
  editing generated model or benchmark source;
- fabricated, incomplete, mismatched-corpus and output-mismatch results fail;
- simulated results are labelled simulated and never populate physical costs.

## R3 — First physical performance evidence

Priority: after R1 and R2.

Use at least one exact Cortex-M4 board and one ESP32-S3 board obtained through a
contributor, lab, reviewer or remote hardware runner. Ownership is not
required.

For each representative model, collect:

- portable BakeNN;
- applicable target BakeNN kernels;
- TFLM built with the same model semantics and the target's normal optimized
  kernel path;
- final Flash, full peak SRAM, initialization cycles, median/p95 inference
  cycles and byte/LSB output comparison.

Initial model set:

- TinyCNN/MNIST-like classifier;
- depthwise-separable MobileNet block;
- residual depthwise block;
- short Conv1D audio/sequence classifier.

Acceptance:

- raw artifacts are reproducible and pass the offline comparison schema;
- regressions and losses are published, not only wins;
- target kernels become measured defaults only for workload regions supported
  by evidence.

## R4 — Target kernel expansion

Priority: only after R3 identifies actual bottlenecks.

### Arm DSP family

- add separate `cortex-m7` and `cortex-m33-dsp` target descriptors;
- do not enable the family for an M33 implementation whose descriptor lacks
  the optional DSP extension;
- generalize the Cortex-M4 SMLAD implementation to an `arm-dsp` capability
  shared by compatible Cortex-M4, Cortex-M7 and DSP-enabled Cortex-M33 targets;
- preserve separate measured-cost tables because M7 cache/TCM and M33 memory
  systems can produce different results from M4 despite sharing instructions;
- record exact code/weight placement such as Flash, SRAM or TCM in benchmark
  evidence;
- add channel/block tails and alignment variants only when measurements show a
  useful workload region.

Planned Arm DSP kernel order:

1. `arm_dsp.linear_smlad.v1`;
2. `arm_dsp.conv2d_1x1_smlad.v1`;
3. `arm_dsp.depthwise_3x3_smlad.v1`;
4. `arm_dsp.conv2d_3x3_im2col_smlad.v1`;
5. `arm_dsp.global_average_pool2d_s8.v1`;
6. `arm_dsp.max_pool2d_2x2_s2.v1`;
7. measured Add/Requantize/Clamp loop variants.

The existing Cortex-M4 IDs remain immutable. The new `arm_dsp.*` IDs are new
implementations even when they initially share source. Cross tests must cover
`-mcpu=cortex-m7` and `-mcpu=cortex-m33`, prove the expected DSP instructions,
and prove that the non-DSP M33 capability path rejects or falls back.

### ESP32-S3

- add explicit Xtensa LX7/vector feature predicates, constant/scratch
  alignment and internal-memory placement metadata;
- implement or integrate a byte-exact vector 1x1 Conv;
- then depthwise 3x3 and Linear;
- then general 3x3 Conv with statically planned scratch;
- then Add/Mul/Requantize/Clamp and pool loops if profiling justifies them;
- choose explicitly between standalone BakeNN intrinsics/assembly and an
  optional ESP-NN vendor backend;
- if ESP-NN is used, keep it optional and verify its rounding/profile contract
  rather than assuming TFLM compatibility implies BakeNN compatibility.

Planned ESP32-S3 kernel order:

1. `esp32s3.conv2d_1x1_vec.v1`;
2. `esp32s3.depthwise_3x3_vec.v1`;
3. `esp32s3.linear_vec.v1`;
4. `esp32s3.conv2d_3x3_tiled_vec.v1`;
5. `esp32s3.add_requantize_clamp_vec.v1`;
6. `esp32s3.global_average_pool2d_s8.v1`;
7. `esp32s3.max_pool2d_2x2_s2.v1`.

The first version may support narrow shape/alignment predicates. `AUTO` must
fall back for unaligned channels, unsupported tails, PSRAM placement or scratch
pressure unless a separately tested variant covers the case. Board evidence
must distinguish internal SRAM from PSRAM and record cache configuration.

### ESP32-P4

- add an `esp32p4` target descriptor rather than treating P4 as an S3 variant;
- declare its RISC-V ISA and PIE/QACC SIMD features explicitly;
- add the current ESP-IDF toolchain, flags, alignment and internal/external
  memory metadata;
- reuse semantic plans and Q31 helpers, but do not reuse S3 Xtensa packed
  layouts or assembly IDs;
- implement P4-native packing and kernels behind separate capability records;
- support optional ESP-NN lowering only as a separately versioned vendor
  backend with a proven arithmetic contract.

Planned ESP32-P4 kernel order:

1. `esp32p4.conv2d_1x1_pie.v1`;
2. `esp32p4.depthwise_3x3_pie.v1`;
3. `esp32p4.linear_qacc.v1`;
4. `esp32p4.conv2d_3x3_tiled_pie.v1`;
5. `esp32p4.add_requantize_clamp_pie.v1`;
6. `esp32p4.global_average_pool2d_s8.v1`;
7. `esp32p4.max_pool2d_2x2_s2.v1`.

P4 acceptance additionally requires disassembly checks for the intended
PIE/QACC instructions and separate measurements for internal versus external
memory. Vendor-published ESP-NN figures may guide prioritization but cannot
populate BakeNN's measured cost table.

### Later targets

- implement ESP32-P4 after the S3 backend contract and common ESP benchmark
  result format are stable;
- do not create RV32IMC or Cortex-M0+ kernels merely to have target-specific
  names: without useful DSP/vector instructions, improve portable C and code
  size first;
- add a RISC-V optimized family only for a selected chip with a declared DSP or
  vector extension and an available toolchain/measurement path.

Acceptance for every kernel:

- versioned implementation and packing-layout IDs;
- exact capability predicate and portable fallback;
- hand goldens, INT8 extremes, asymmetric zero points, positive/negative
  requantization shifts and accumulator-bound cases;
- randomized Python/portable/target differential tests;
- strict compiler, sanitizer where executable, cross-link, symbol and
  disassembly checks;
- physical measurements before enabling it as a measured default.

## R5 — Model and operator maturity

Priority: parallel with measurement infrastructure where it does not alter the
numeric core.

Vision track:

- **Host baseline complete:** MobileNetV3-small, EfficientNet-Lite-style and
  compact U-Net model surfaces compile, not only their individual operations;
- close capture gaps discovered by those full graphs;
- add model-level FP32-versus-dequantized-INT8 accuracy reports;
- add layout/constant deduplication and fusion based on real model artifacts.

Audio track:

- compile a complete DS-CNN/keyword-spotting style Conv1D model;
- define a documented NLC firmware input contract for PCM/features;
- add streaming-window examples while keeping each compiled invocation static;
- add model-level accuracy and generated-C tests.

### MobileNetV3 completion

Status: implemented and host-tested; dataset/target evidence remains.

MobileNetV3 is the nearest full-model expansion. Most of its compute surface is
already present. The main missing semantic is static channel broadcast in the
SE path:

```text
[1, H, W, C] * [1, 1, 1, C] -> [1, H, W, C]
```

Implemented:

- extend Add/Mul verification and lowering with an explicit static broadcast
  map; never rely on implicit C pointer arithmetic;
- support deterministic same-rank static broadcast maps, including NHWC SE
  channel broadcast;
- prove Q31 bounds under reuse of the broadcast operand;
- add portable-C and integer-reference kernels, followed by producer/clamp
  fusion where rounding points permit it;
- compile an unmodified eval-mode MobileNetV3-small reference graph through
  capture, PTQ, planning and generated C;
- report FP32, dequantized INT8-reference and generated-C accuracy on a fixed
  public evaluation subset.

Acceptance:

- the complete model has no manually replaced SE block;
- Python integer and generated C are byte-exact;
- unsupported broadcast axes fail with the exact offending shapes;
- Flash/SRAM fit and latency remain measurements, not assumptions.

The compact MobileNetV3 gate runs generated C under GCC/Clang sanitizers and is
byte-exact with the integer reference. An unmodified torchvision
`mobilenet_v3_small` and `mobilenet_v3_large` at static 32x32 reach C artifact generation. A fixed
public pretrained evaluation subset is still required before publishing an
accuracy claim.

### EfficientNet-Lite completion

Status: implemented and host-tested for the declared Lite-style surface;
dataset/target evidence remains.

Google's EfficientNet-Lite design replaces SiLU with ReLU6 and removes SE.
BakeNN compiles that MBConv surface. As a stronger frontend stress case, an
unmodified torchvision EfficientNet-B0 graph retaining SiLU and SE broadcast
also reaches generated C artifacts at static 32x32.

Implemented:

- reuse the verified broadcast contract from MobileNetV3;
- normalize framework SAME padding into explicit static padding;
- close adaptive-pool, squeeze and reshape variants found in the selected
  exported model;
- remove eval-only stochastic-depth/dropout semantics during capture;
- compile one exact static input resolution; other resolutions are separate
  compile-time shapes rather than a dynamic runtime contract;
- add full-model arena/liveness and PTQ accuracy reports.

Acceptance:

- documentation names the exact model revision and resolution;
- no operator is accepted through a silent semantic approximation;
- generated C is byte-exact with the integer reference for the full graph.

### U-Net and encoder-decoder models

Status: portable implementation and compact model gates complete; target fit
and optimized/tiled kernels remain.

U-Net is technically supported by extending tensor transforms, but its long
skip lifetimes make SRAM a first-class acceptance constraint.

Implemented portable kernel order:

1. `ResizeNearest2D` with identical input/output qparams;
2. fixed-point `ResizeBilinear2D` with a versioned rounding profile;
3. direct or lowered `ConvTranspose2D`;
4. measured tiled variants remain deferred until physical profiling.

Implemented:

- define exact coordinate transformation and edge rules for each Resize mode;
- choose and version bilinear coefficient precision and tie rounding;
- implement ConvTranspose verifier, accumulator proof, reference and portable
  C, including stride, padding, output padding and per-channel qparams;
- compare direct scatter accumulation with zero-insert-plus-Conv lowering
  without changing numeric semantics;
- retain skip tensors through liveness and reject models whose declared SRAM
  budget cannot hold the activation plan;
- compile a fixed-resolution compact U-Net fixture end to end.

Grouped ConvTranspose2D, static-crop and bilinear/nearest-resize U-Net variants
run as strict sanitized C and match the integer reference byte-for-byte.
Shapes, slice bounds, resize maps, skip liveness and arena offsets are frozen at
host compile time. Runtime-dependent slice/resize sizes remain unsupported.

Acceptance:

- Resize/ConvTranspose hand goldens cover corners, odd dimensions and
  asymmetric padding;
- no temporary allocation occurs at runtime;
- the manifest exposes the SRAM cost of retained skip tensors and scratch;
- a model that does not fit fails during host compilation.

### YOLO and SSD detection

Detection is split into two products:

```text
model.c          INT8 backbone and raw detection head
postprocess.c    decode, threshold, TopK and fixed-capacity NMS
```

The first milestone supports raw heads. It must name one exact model version;
`YOLO support` is not a single stable operator contract.

Raw-head work:

- add the remaining static Transpose/Reshape semantics
  required by the selected model;
- support its exact activation and head decoding inputs without dynamic shape;
- expose fixed-shape quantized head tensors or feed them directly to generated
  postprocessing;
- compile and compare the complete backbone/head before implementing NMS.

Postprocessing work:

- define a fixed `MAX_DETECTIONS` output ABI;
- implement versioned anchor/grid or distribution decode for the selected
  YOLO/SSD revision;
- implement deterministic threshold, TopK and NMS ordering, including tie
  behavior;
- choose explicitly between fixed-point coordinates and an optional
  application-side float postprocessor;
- bound every candidate array statically and reject configurations that exceed
  the declared SRAM budget;
- test empty, all-overlapping, equal-score and maximum-capacity cases.

Acceptance:

- raw neural-network output remains independently testable;
- `postprocess.c` has no heap or unbounded output;
- model/version, anchors, thresholds, coordinate profile and maximum detection
  count are recorded in the manifest;
- detection quality is evaluated against the original model, not only through
  C-versus-Python agreement.

Likely next operators should be admitted only when a target model needs them:

- general adaptive pooling or additional reduction axes;
- transpose/layout transforms that cannot remain host-only;
- fixed TopK/NMS support for one declared detection model revision;
- additional activation profiles where a 256-byte LUT is not the right trade.

This phase does not mean following the entire TFLM operator catalog. Each new
operator requires verifier, lowering, overflow proof, reference, portable C,
frontend mapping and malformed/randomized tests.

## R6 — Fully quantized TFLite import

Priority: after the core selection and target evidence contracts are stable.

Purpose: accept `.tflite` as a host-side interchange format without shipping a
TFLite interpreter or FlatBuffer parser in firmware.

Work:

- make FlatBuffer parsing an optional host dependency;
- import static fully-quantized INT8 tensors and supported operator semantics
  into `QuantizedGraph`;
- validate tensor layouts, per-axis qparams, bias scales, fused activation and
  operator versions;
- define how TFLite's arithmetic behavior maps to BakeNN profiles;
- reject an arithmetic mismatch unless a separately versioned, tested profile
  exists;
- add TFLite interpreter versus BakeNN integer/C differential fixtures.

Acceptance:

- generated firmware has no TFLite, FlatBuffers or interpreter dependency;
- unsupported operators and incompatible qparams fail at import time;
- supported imports have byte-exact or explicitly documented profile-level
  comparison results.

## R7 — Optional QAT

Priority: after PTQ model coverage identifies accuracy failures that
calibration improvements cannot solve.

Work:

- implement PyTorch fake-quant modules using the same activation/weight ranges
  and rounding intent as the deployment contract;
- support observer freeze, qparam freeze and conversion to the existing
  `QuantizedGraph`;
- keep training APIs separate from IR/backend APIs;
- compare FP32, PTQ and QAT accuracy on representative models;
- after conversion, require the same integer-reference/C byte-exact gates as
  PTQ.

QAT must not introduce a second deployment arithmetic implementation. It is a
training path that ends at the existing verified quantized IR.

## R8 — Compiler and release hardening

Compiler work:

- implement profitable whole-tensor constant folding rather than analysis-only
  reporting;
- expand safe constant and packed-representation deduplication;
- add model-level loop fusion only when integer rounding points are preserved;
- improve static arena planning for larger branched graphs;
- record why every proposed fusion or target kernel was accepted or rejected;
- add an optional generated-source split mode if one large `model.c` becomes a
  compiler or incremental-build problem.

Release work:

- freeze the public Python API and generated C ABI for the first release;
- test wheel build, clean install, PyTorch optional extra and cross-toolchain
  matrix in CI;
- publish complete generated examples and reproducible benchmark bundles;
- add semantic-version rules for IR, arithmetic profile, kernel ID, packing
  layout, manifest and result schema;
- document supported models separately from supported individual operations.

## Explicitly deferred

These are not near-term goals:

- dynamic shapes or dynamic batch;
- arbitrary multi-input/multi-output graphs;
- on-device training or dynamic quantization;
- automatic float fallback;
- full transformer support, KV cache or large language models;
- complete parity with every TFLM operation;
- NPU code generation before a specific NPU and deployment contract is chosen.

## Recommended execution order

1. R1 measured-selection semantics.
2. R1A OPT-01/02: establish the fair benchmark and remove avoidable overhead
   from the generic portable Conv.
3. R1A OPT-03/04: add the static direct and budgeted-unroll candidates behind
   explicit experimental/static selection.
4. R2 remote hardware benchmark bundles and validators.
5. R3 obtain one Cortex-M4 and one ESP32-S3 physical result set, including the
   portable and eligible R1A Conv candidates.
6. Populate only evidence-backed cost entries and enable measured selection.
7. R4 generalize Arm DSP for Cortex-M7/M33, then implement ESP32-S3 and
   ESP32-P4 kernels in the declared order, guided by measured bottlenecks.
8. R5's MobileNetV3-small, EfficientNet-Lite and compact U-Net host baselines
   are complete; next admit one exact YOLO/SSD raw-head and postprocess
   contract only after its static ABI is specified.
9. R6 TFLite import.
10. R7 QAT if PTQ accuracy data justifies it.
11. R8 API/ABI freeze and release evidence.

The next implementation task should therefore be R1 selection semantics,
followed by the R1A benchmark and portable-Conv cleanup—not another
speculative target kernel or an unmeasured claim based on host timing.
