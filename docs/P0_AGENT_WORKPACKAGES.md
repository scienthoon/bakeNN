# P0 Agent Work Packages

This file divides P0 into dependency-ordered, reviewable changes. Agents must
not begin a package before its dependencies are merged. Each package must keep
all existing tests green and add its own acceptance tests.

## Ownership rule

Parallel agents receive non-overlapping operator-family modules and tests.
Central registries, exports, and compiler orchestration are changed only by the
integration owner. No agent may rewrite another package's files without an
explicit handoff.

## WP-00: Generic compiler substrate

Dependency: current Linear-only MVP

Owner scope:

```text
src/bakenn/ir/op.py
src/bakenn/ir/graph.py
src/bakenn/ir/verify.py
src/bakenn/plan/types.py
src/bakenn/plan/memory.py
src/bakenn/plan/lower.py
src/bakenn/reference/executor.py
tests/p0/test_generic_graph.py
tests/p0/test_generic_memory.py
```

Tasks:

- Convert the Linear-specific graph walk to immutable generic op inputs/outputs.
- Convert flat `plan.py` and `reference.py` into packages while preserving the
  current public API and all MLP tests.
- Add multi-input liveness, alias-group representation, scratch accounting, and
  generic step dispatch.
- Keep Linear as the first registered op family.
- Add fail-closed checks for dead ops, multiple producers, cycles/use-before-
  definition, missing constants, overlapping live buffers, and unsafe aliases.

Acceptance:

- Existing 12 tests pass unchanged or through compatibility imports.
- A synthetic diamond graph keeps both branches alive through a join.
- A four-stage graph demonstrates safe arena reuse after last use.
- Graph/plan constants and mappings remain immutable snapshots.

## WP-10: Conv family vertical slice

Dependency: WP-00

Owner scope:

```text
src/bakenn/ir/ops/conv.py
src/bakenn/ir/verifiers/conv.py
src/bakenn/plan/lowering/conv.py
src/bakenn/reference/kernels/conv.py
src/bakenn/backend/portable_c/kernels/conv.py
tests/p0/test_conv_ir.py
tests/p0/test_conv_c.py
```

Tasks:

- Implement Conv2D and DepthwiseConv2D contracts from the blueprint.
- Implement output-shape validation, per-channel qparams, accumulator and
  positive-shift proofs, input-zero-point padding, fused clamp, integer
  reference, and portable C emission.
- Cover stride, asymmetric padding, dilation, depth multiplier, nonzero input
  zero point, per-channel multipliers, saturation, and rejection cases.

Acceptance:

- Hand goldens and at least 256 randomized inputs are Python/C bit-exact.
- Generated kernels pass strict GCC/Clang, ASan, and UBSan.
- Tests prove padding uses input zero point and int32 overflow is rejected.

## WP-20: Elementwise, activation, pooling, and shape family

Dependency: WP-00

Owner scope:

```text
src/bakenn/ir/ops/elementwise.py
src/bakenn/ir/ops/pool.py
src/bakenn/ir/ops/shape.py
src/bakenn/ir/verifiers/elementwise.py
src/bakenn/ir/verifiers/pool.py
src/bakenn/ir/verifiers/shape.py
src/bakenn/plan/lowering/elementwise.py
src/bakenn/plan/lowering/pool.py
src/bakenn/plan/lowering/shape.py
src/bakenn/reference/kernels/elementwise.py
src/bakenn/reference/kernels/pool.py
src/bakenn/reference/kernels/shape.py
src/bakenn/backend/portable_c/kernels/elementwise.py
src/bakenn/backend/portable_c/kernels/pool.py
src/bakenn/backend/portable_c/kernels/shape.py
tests/p0/test_elementwise_c.py
tests/p0/test_pool_c.py
tests/p0/test_shape_c.py
```

Tasks:

- Implement Add, Mul, Clamp/ReLU/ReLU6, internal Requantize, Avg/Max Pool,
  Reshape/Flatten, and Concatenate.
- Add view alias groups and legal output routing.
- Add the explicit activation-fusion pass; never fuse across fan-out.
- Insert Requantize before Concatenate when qparams differ.

Acceptance:

- Residual diamond liveness is correct and canaries remain intact.
- AveragePool non-power-of-two windows and negative ties are golden-tested.
- Reshape emits no kernel/copy for internal views.
- Concatenate works on NHWC channel and NC feature axes and rejects unsafe
  in-place schedules.

## WP-30: Softmax family

Dependency: WP-00

Owner scope:

```text
src/bakenn/ir/ops/softmax.py
src/bakenn/ir/verifiers/softmax.py
src/bakenn/plan/lowering/softmax.py
src/bakenn/reference/kernels/softmax.py
src/bakenn/backend/portable_c/kernels/softmax.py
tests/p0/test_softmax_c.py
```

Tasks:

- Implement the declared Q15 LUT profile and fixed output qparams.
- Prove LUT sum range, handle all-equal inputs, large negative differences,
  saturation, and deterministic constant emission.
- Expose the arithmetic profile in the manifest.

Acceptance:

- Python/C bit-exact for exhaustive int8 inputs when class count is two and
  randomized vectors for class counts 3, 10, and 100.
- Output probability codes stay in `[0,255]`, all-equal rows are symmetric to
  within one code, and no divide-by-zero path exists.

## WP-40: PyTorch FP32 frontend and PTQ

Dependencies: WP-10, WP-20, WP-30

Owner scope:

```text
src/bakenn/frontends/torch_export/
src/bakenn/quantization/observers.py
src/bakenn/quantization/ptq_graph.py
src/bakenn/passes/batchnorm.py
src/bakenn/passes/layout.py
src/bakenn/passes/legalize.py
src/bakenn/passes/fuse.py
tests/p0/test_torch_frontend.py
tests/p0/test_ptq_graph.py
```

Tasks:

- Import the exact P0 PyTorch operation subset from `torch.export`.
- Fold eval BatchNorm, remove Dropout/Identity, canonicalize layouts, observe
  graph edges, choose qparams, quantize constants, insert Requantize, and fuse
  activation clamps.
- Reject training-mode modules, dynamic dimensions, unsupported grouped conv,
  broadcasting, unsupported axes, and unsupported aten operators.

Acceptance:

- TinyCNN, residual DS-CNN, and Mobile block lower to verified QuantizedGraph.
- Calibration is deterministic under a fixed model/data/seed.
- FP32-to-dequantized-INT8 error is reported per output and per observed edge.

## WP-50: Integration, generated ABI, and competitive benchmark

Dependencies: WP-10, WP-20, WP-30, WP-40

Owner scope:

```text
src/bakenn/compiler.py
src/bakenn/backend/portable_c/emitter.py
src/bakenn/report.py
tests/p0/test_models_c.py
benchmarks/tflm_compare/
docs/P0_STATUS.md
```

Tasks:

- Wire all op families without framework types crossing the frontend boundary.
- Emit exact layout/qparam/resource macros and deterministic manifest.
- Run whole-model Python/C differential and sanitizer tests.
- Add reproducible TFLM comparison instructions and result schema.

Acceptance:

- All release gates in `P0_BLUEPRINT.md` pass.
- No generated artifact references heap functions or C++ symbols.
- A clean wheel installation can compile all three fixtures without source-tree
  imports or network access.

## Agent handoff format

Every implementing agent must report:

```text
Files changed
Contract decisions made
Tests added and exact commands run
Known exclusions
Anything the next package must not assume
```

An agent may not weaken a contract to make a test pass. Ambiguities are resolved
in the blueprint first, then implementation continues.
