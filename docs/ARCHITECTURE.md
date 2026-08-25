# Architecture

The compiler separates model meaning from target execution details.

```text
PyTorch frontend -> FP32 graph -> calibration/PTQ -> QuantizedGraph
                                                   |
              optional quantized TFLite importer ---+
                                                   v
                         verification and legalization
                                                   |
                                                   v
ExecutionPlan        schedule, fixed-point parameters, storage offsets
      |                         |
      v                         v
integer reference         C backend + explicit kernel policy
```

## Dependency rules

- Frontends may depend on their source framework and must immediately convert
  framework objects into BakeNN-owned types.
- The IR, planner, reference executor, and C backend never import PyTorch,
  TensorFlow, TFLite, or FlatBuffers.
- A backend consumes only an `ExecutionPlan`; it does not infer quantization or
  memory placement.
- The post-lowering reference executor and generated C must be bit-exact.
- There is no float fallback after quantization.

The PyTorch PTQ path returns both the deployed artifacts and an immutable
`calibration_report` containing the consumed sample count, observed FP32 edge
ranges, and chosen activation qparams. Its `verify_accuracy()` method runs a
separate validation corpus through FP32 evaluation and the deployed integer
plan and reports per-layer maximum/mean/RMS error plus INT8 endpoint counts.
This is an accuracy diagnostic; generated C correctness is still gated by
byte-exact comparison with the integer reference.

## Backend selection boundary

Kernel selection is a host compilation decision. `PORTABLE` is the default;
`STATIC_PRIORITY` chooses the highest-priority candidate whose capability
predicate holds; and `MEASURED` uses a physical cycle entry only when its
canonical workload, exact target, toolchain, and compiler flags match. Without
such an entry, `MEASURED` selects portable C. `AUTO` is a deprecated alias for
static priority and never means measured fastest. Selection basis, rejected
candidates, packed layouts, and any matched cost evidence are recorded in the
manifest.

## Storage model

Graph inputs and outputs are caller-owned. Constants live in read-only storage.
Intermediate activations receive aligned offsets in one caller-owned arena.
The liveness planner does not overlap buffers that are live during the same
operation. Future in-place transformations must be explicit plan rewrites.

## Implemented v1 P0 vertical slice

1. Generic immutable SSA IR, fail-closed per-op verification, and graph-edge qparams
2. Conv/Depthwise/Linear/ConvTranspose, static-broadcast elementwise,
   activation, pool, resize, static slice/crop, shape, residual, and Softmax families
3. Multi-input liveness, alias groups, reusable activation arena, and max scratch plan
4. Bit-exact Python integer reference and model-specialized portable C11 backend
5. Real `torch.export` capture plus deterministic PTQ primitives and graph passes

Optimized target overlays, cross-toolchain verification, and scoped physical
evidence now exist, while further model-driven operator coverage, broader
physical-target collection, and QAT remain roadmap work. The R6 host-only
TFLite importer immediately converts its strict, fully-quantized subset into
`QuantizedGraph`. The optional `tflite`/FlatBuffers packages remain frontend
dependencies only: the generated target has no TFLite runtime or FlatBuffers
dependency.

The importer currently accepts TFLite schema version 3 with one subgraph, one
public input/output, static batch-one rank-2/3/4 INT8 activations, and these
exact builtin versions: `CONV_2D` v3, `DEPTHWISE_CONV_2D` v3,
`FULLY_CONNECTED` v4 with bias or v6 without bias, `ADD` v2,
`AVERAGE_POOL_2D`/`MAX_POOL_2D` v2, `RESHAPE` v1, and `PAD`/`PADV2` v2. Only
fused `NONE`, `RELU`, and `RELU6` are accepted. Dynamic/variable/sparse tensors,
external buffers, custom ops, unsupported versions, incompatible qparams,
grouped Conv2D, and other builtins fail during import. Weights must be symmetric
INT8 with zero point 0 and the expected per-axis dimension; INT32 bias scales
must equal input scale times weight scale, and padding must encode affine real
zero.

Importer differential tests execute the same model with LiteRT's built-in
reference resolver, BakeNN's integer reference, and generated C. The three
outputs must be byte-exact. Optimized host delegates are deliberately not the
oracle because their rounding ties need not identify the target firmware
profile.

TFLite `AVERAGE_POOL_2D` uses the separately versioned
`tflite.int8.average_pool2d.raw_code.v1` profile: it rounds the mean of raw INT8
codes, matching LiteRT `BUILTIN_REF`. BakeNN's native
`bakenn.int8.average_pool2d.v1` profile instead averages values centered around
the zero point. These expressions can differ by one LSB at a half tie when the
zero point is nonzero, so the importer never silently maps one profile to the
other. The benchmark-only BakeNN-to-TFLite serializer likewise rejects a
native-profile AveragePool instead of claiming a byte-exact round trip.

QAT is intentionally downstream of the integer contract: it adapts training to
the deployment arithmetic but does not redefine backend arithmetic.
