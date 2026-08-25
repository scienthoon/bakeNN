# BakeNN target layer

Status: ARM/RISC-V cross-link path host-verified; ESP-IDF packaging implemented;
physical nRF52840/Cortex-M4 FC and standalone Conv evidence plus original-ESP32
trained MNIST and MobileNetV2 evidence recorded; other targets and full
peak-resource claims remain explicitly scoped or unmeasured

BakeNN keeps numerical semantics separate from target lowering:

```text
QuantizedGraph
    -> ExecutionPlan                 target independent, bakenn.int8.v1
    -> CBackendPlan + TargetDescriptor
    -> C11 sources / packed constants
    -> target ELF or ESP-IDF component
```

`TargetDescriptor` records ISA, ABI, features, storage alignment, compiler
flags, optional board resource budgets, and explicitly measured kernel costs.
It may not change tensor qparams, rounding, operator order, or liveness.

## Built-in profiles

| Profile | ISA/toolchain | Current evidence |
|---|---|---|
| `portable32` | generic C11 ILP32 | GCC/Clang execution and sanitizer tests |
| `cortex-m0plus` | ARMv6-M, GNU Arm Embedded | freestanding ELF link and symbol audit |
| `cortex-m4` | ARMv7E-M/DSP, GNU Arm Embedded | intrinsic-kernel ELF, `smlad` disassembly and symbol audit |
| `rv32imc` | RISC-V I/M/C ILP32, GNU Embedded | freestanding ELF link and symbol audit |
| `esp32` | Xtensa LX6, ESP-IDF | CI build plus physical trained-MNIST and MobileNetV2 evidence |
| `esp32s3` | Xtensa LX7/vector, ESP-IDF | ESP-NN component/project CI; no physical result yet |
| `esp32c3` | ESP RISC-V, ESP-IDF | portable-fallback component/project CI |

ESP-NN is optional and selected only when explicitly enabled with a non-portable
kernel policy and all capability predicates hold. Original ESP32 currently uses
optimized ESP-NN Conv/Depthwise paths while unsupported FC/pool cases fall back
to portable C. ESP32-S3 can select its pinned Conv, Depthwise, per-channel FC
and pooling source closures; ESP32-C3 intentionally remains portable because
the pinned ESP-NN revision has no matching optimized C3 contract. Every fallback
preserves the same verified INT8 semantics and is recorded in the manifest.

The Cortex-M4 profile can select versioned `SMLAD` Linear, 1x1 Conv, depthwise
3x3 and im2col 3x3 Conv kernels plus specialized global-average and 2x2
max-pool loops. Cross-compilation proves the instructions and link contract;
only a physical run can establish cycles, cache/bus effects, stack watermark,
or energy. Other profiles currently use generic or portable kernels.

## Compilation

Select a built-in target by id:

```python
compiled = bakenn.compile(graph, "build/model", target="cortex-m4")
```

Or declare exact board capacities without pretending that a whole CPU family
has one memory size:

```python
from dataclasses import replace
from bakenn.targets import CORTEX_M4

board = replace(CORTEX_M4, flash_bytes=512 * 1024, sram_bytes=128 * 1024)
compiled = bakenn.compile(graph, "build/model", target=board)
```

The host compiler rejects a constant payload larger than declared Flash and an
arena larger than declared SRAM. These are necessary lower-bound checks, not
full firmware fit proofs: code, initialized data, other application globals,
interrupt stacks, and task stacks require the final ELF/map and runtime stack
measurement.

## Freestanding ARM/RISC-V verification

```python
report = bakenn.build_freestanding_elf(
    compiled.artifacts,
    board,
    "build/model/cross",
)
```

The verifier uses the target's GNU embedded compiler and:

- compiles strict freestanding C11;
- links one final ELF and linker map;
- links compiler integer-runtime helpers when required;
- rejects unresolved symbols;
- rejects heap allocation and software floating-point helper symbols;
- records linked text/data/bss, Flash-load and static-SRAM bytes;
- leaves stack and cycles explicitly unmeasured.

The synthetic linker addresses are for compatibility/resource inspection, not
a board startup image. A production firmware must use the board vendor's
startup code and linker script.

## ESP-IDF packaging

```python
compiled = bakenn.compile(graph, "build/model", target="esp32s3")
project = bakenn.export_esp_idf_project(
    compiled.artifacts,
    "esp32s3",
    "build/esp32s3_project",
)
```

The project contains a `bakenn_model` component and a runner that prints the
cycle counter, FreeRTOS stack high-water mark, arena size, and output bytes when
it is eventually flashed. A boardless `idf.py build` proves toolchain/link
compatibility and final firmware section sizes; it cannot prove latency,
energy, cache behaviour, or physical peak stack usage.

## Measured cost tables

`KernelCostMeasurement` accepts only positive cycle counts with a kernel id,
workload, exact toolchain/flags, and evidence reference. All built-in targets
currently contain zero entries.

`PORTABLE` is the default and excludes optimized candidates.
`STATIC_PRIORITY` is an explicit capability-priority mode: it chooses the
highest-priority supported candidate but makes no physical performance claim.
`MEASURED` considers a cost only when kernel ID, canonical workload, exact
target descriptor, toolchain, and compiler flags match; if no exact entry is
available, it selects portable C. `AUTO` is deprecated and retained only as an
alias for `STATIC_PRIORITY`; it is not a fastest-kernel mode.

Never copy host timings, datasheet estimates, or another MCU's numbers into a
target cost table. Physical target measurements are the gate for enabling a
target-specific kernel by default.

Cross-building an ELF or ESP-IDF project proves only the declared compiler,
link, symbol, and section-size properties. Physical cycle, latency, stack, and
energy evidence stays in the separate physical-board index and is scoped to
the exact board, firmware, toolchain, flags, and workload; cross-build records
must not be promoted into a measured cost table.
