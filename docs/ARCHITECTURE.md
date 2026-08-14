# Architecture

The compiler separates model meaning from target execution details.

```text
frontends / PTQ
      |
      v
QuantizedGraph       static tensor types and quantization semantics
      |
      v
verification         fail closed on unsupported or unsafe models
      |
      v
ExecutionPlan        schedule, fixed-point parameters, storage offsets
      |                         |
      v                         v
integer reference         portable C backend
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

## Storage model

Graph inputs and outputs are caller-owned. Constants live in read-only storage.
Intermediate activations receive aligned offsets in one caller-owned arena.
The liveness planner does not overlap buffers that are live during the same
operation. Future in-place transformations must be explicit plan rewrites.

## Implemented P0 vertical slice

1. Generic immutable SSA IR, fail-closed per-op verification, and graph-edge qparams
2. Conv/Depthwise/Linear/ConvTranspose, static-broadcast elementwise,
   activation, pool, resize, static slice/crop, shape, residual, and Softmax families
3. Multi-input liveness, alias groups, reusable activation arena, and max scratch plan
4. Bit-exact Python integer reference and model-specialized portable C11 backend
5. Real `torch.export` capture plus deterministic PTQ primitives and graph passes

Post-P0 work includes the optimized target backend, further model-driven
operator coverage, physical-target resource/latency collection, and QAT. Fully-quantized
TFLite or ONNX may later be accepted as optional input formats, but neither
becomes a runtime or compiler-core dependency.

QAT is intentionally downstream of the integer contract: it adapts training to
the deployment arithmetic but does not redefine backend arithmetic.
