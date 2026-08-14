# BakeNN versus TFLM benchmark protocol

This directory is an offline harness **skeleton**. It does not download, install,
or vendor TFLite Micro, and `example_result.json` contains no claimed benchmark
numbers. A result is publishable only after both implementations are built for
the same physical target and execute the same quantized model semantics and
input bytes.

## Fair-comparison contract

1. Freeze one fully quantized model. Record a SHA-256 over its canonical BakeNN
   QuantizedGraph artifact and the exact `.tflite` FlatBuffer used by TFLM. The
   two artifacts must use the same INT8 weights, biases, tensor qparams,
   padding, fused clamps, and rounding profile. If translation changes an
   arithmetic profile (for example BakeNN Softmax versus TFLite Softmax), report
   the models separately; do not call them the same model.
2. Generate one binary input corpus of raw INT8 model-input bytes. Feed it to
   both binaries in identical order and record its SHA-256 and input count.
3. Use the same board, MCU clock, memory wait states/cache state, compiler
   executable and version, optimization/LTO flags, linker script, startup code,
   cycle-counter implementation, interrupt policy, warmups, and measured runs.
4. Register only the TFLM operators used by the model. Record whether reference,
   CMSIS-NN, or another optimized kernel library is selected. Compare BakeNN's
   matching portable or optimized backend; label the backend in `version`.
5. Build separate minimal firmware images whose non-ML scaffolding is
   byte-identical where practical. Never subtract an estimated runtime size
   from either ELF after linking.

## Required provenance

Every JSON result records:

- BakeNN version and source commit;
- TFLM version/source commit and kernel backend;
- model and input-corpus SHA-256 identifiers;
- board revision, exact MCU, ISA/extensions, and clock;
- compiler name, complete version, every compile/link flag, and linker-script
  identity;
- warmup/measured run counts, cycle counter, and interrupt policy.

Replace all `replace-with-*` strings before publishing a measured result.

## Exact metrics

All byte and cycle metrics are non-negative integers. Each metric is an object
with either `{"status":"measured","value":N,"reason":null}` or
`{"status":"unmeasured","value":null,"reason":"..."}`. Estimates must not be
reported as measurements.

### Final ELF Flash

Read the final linked target ELF/map, after dead stripping and with production
flags. Report:

- `elf_flash_text_bytes`: executable `.text` and target-equivalent executable
  Flash sections;
- `elf_flash_rodata_bytes`: read-only model/runtime constants in Flash;
- `elf_flash_data_load_bytes`: Flash load image for initialized writable data;
- `elf_flash_total_bytes`: exact sum of the preceding three fields.

Record the exact section-to-category mapping for a target beside the result.
Do not use generated C file size, object size before linking, or the BakeNN
manifest's `constant_bytes` as final ELF Flash.

### Peak SRAM

Measure simultaneous peak target SRAM, including:

- `peak_sram_arena_bytes`: BakeNN activation/scratch arena or TFLM tensor arena;
- `peak_sram_runtime_metadata_bytes`: interpreter, allocator, tensor/node,
  registration, and persistent operator metadata. BakeNN may measure zero only
  when the final map/instrumentation proves none exists;
- `peak_sram_stack_bytes`: per-implementation stack high-water mark under the
  inference protocol;
- `peak_sram_static_data_bytes`: writable `.data` plus `.bss` not counted in
  arena or stack;
- `peak_sram_total_bytes`: exact sum of those four disjoint components.

Use linker-map accounting for static allocations and a target-tested stack
watermark. For TFLM, include `MicroInterpreter`, runtime tensors and persistent
tail allocations; reporting only `tensor_arena_size` is invalid. Avoid double
counting a statically declared arena in both arena and `.bss`.

### Initialization and inference cycles

- `init_cycles`: reset of benchmark-owned state through readiness for the first
  inference. BakeNN includes any explicit initialization; a true no-op is zero.
  TFLM includes model validation, resolver/interpreter construction and
  `AllocateTensors()`.
- `inference_median_cycles`: integer median of individual measured `Invoke()` or
  generated-model calls after warmup.
- `inference_p95_cycles`: nearest-rank 95th percentile of the same raw samples.

Read a hardware cycle counter around only the defined region, subtract a
measured counter-read baseline consistently, define wrap handling, and do not
derive cycles from wall-clock time. Keep raw cycle samples with the result.

### Output bytes and error

For the shared input corpus compare raw output buffers:

- `output_compared_bytes`: total number of bytes compared;
- `output_mismatched_bytes`: positions whose raw INT8 codes differ;
- `output_max_abs_error_lsb`: maximum absolute difference between corresponding
  integer codes, expressed in output-code LSBs.

Byte-identical output has mismatch count and max error zero. If output qparams
or arithmetic profiles differ, first declare the semantic difference and use a
separate real-domain error report; do not disguise it as an identical-model
comparison.

## Workflow

```text
1. Freeze model + input corpus; compute SHA-256 values.
2. Build BakeNN firmware and TFLM firmware with one recorded toolchain config.
3. Extract final-ELF sections and linker-map SRAM components.
4. Flash the same board and collect init/inference raw cycle samples.
5. Capture raw output bytes and compute byte/LSB error metrics.
6. Copy example_result.json, replace provenance, mark each available metric
   measured, and leave genuinely unavailable metrics explicitly unmeasured.
7. Validate locally:

   python benchmarks/tflm_compare/validate_result.py result.json
```

The validator uses only Python's standard library. It checks required fields,
types, measured/unmeasured consistency, Flash/SRAM component sums, percentile
ordering, and output-count invariants. It does not decide whether a physical
measurement was performed honestly; preserve scripts, map files, raw samples,
ELFs, and serial logs as evidence.
