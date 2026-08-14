# BakeNN P0 Blueprint

Status: implemented host baseline; physical-target performance evidence pending

P0 is the first release that can compile representative fixed-shape MCU
networks rather than a demonstration MLP. It is deliberately narrower than
TFLite Micro, but every advertised combination must be statically validated,
integer-exact, memory-planned, and emitted as standalone C11.

## Product boundary

P0 compiles one already-trained, static, batch-one PyTorch inference graph plus
calibration samples into a model-specific C library.

```text
PyTorch FP32 eval model + representative calibration data
                        |
                        v
              torch.export frontend
                        |
                        v
            observed framework-neutral graph
                        |
                        v
              PTQ QuantizedGraph
                        |
             verify / fuse / legalize
                        |
                        v
                 ExecutionPlan
                 /             \
                v               v
      integer reference     standalone C11
```

The generated target artifact contains no PyTorch, TFLite, FlatBuffers,
interpreter, operator registry, C++ runtime, heap allocation, or runtime graph
preparation. A fully-quantized TFLite importer is not part of P0; it will be a
later optional frontend that lowers into the same `QuantizedGraph`.

## Release gates

P0 is complete only when all of the following are true:

1. Three representative graphs compile end-to-end from FP32 PyTorch:
   `TinyCNN`, a residual DS-CNN block, and a MobileNetV1-style depthwise block.
2. The post-quantization Python executor and generated C are bit-exact for all
   supported operators and graph combinations.
3. Unsupported dtype, shape, parameter, or unsafe accumulator fails on the
   host with a deterministic compiler error. There is no float fallback.
4. The manifest reports constants, activation arena, scratch, I/O, alignment,
   and arithmetic profile. Reported target allocations match generated ABI.
5. Generated C passes GCC and Clang with strict warnings, ASan, and UBSan.
6. A benchmark harness can compare the same model against TFLM for final ELF
   Flash, peak SRAM, initialization cycles, inference cycles, and output error.

> Historical baseline: this document freezes the original P0 acceptance
> contract. The implemented surface has since expanded to static broadcast,
> additional activations, Conv1D/grouped Conv, Resize2D and ConvTranspose2D.
> See `P0_STATUS.md` and `ROADMAP.md` for the current contract; exclusions below
> describe the original P0 milestone rather than current BakeNN capability.

## P0 operator surface

The public supported surface is:

| Family | P0 operations |
|---|---|
| Compute | Conv2D, DepthwiseConv2D, FullyConnected |
| Elementwise | Add, Mul |
| Activation | ReLU, ReLU6, represented as an integer clamp and fused when legal |
| Pooling | AveragePool2D, MaxPool2D |
| Shape/data movement | Reshape, Flatten, Concatenate |
| Output | Softmax on the final channel dimension |

The compiler also owns one internal `Requantize` operation. It is inserted to
align qparams for operations such as Concatenate and is not advertised as a
model-layer feature.

BatchNorm is not a target operation. Eval-mode BatchNorm parameters must be
folded into Conv/Depthwise/Linear constants before quantization. Dropout and
Identity must be removed. Grouped convolution is rejected unless `groups=1`
or it is the exact depthwise case `groups=input_channels`.

## Canonical tensor and layout contract

All dimensions are compile-time positive integers. P0 supports one model input,
one model output, and batch size one.

The portable 32-bit target profile limits each dimension to `INT32_MAX` and
each tensor's storage to `UINT32_MAX` bytes. Host verification additionally
proves every generated Conv/Pool signed coordinate expression fits `int64_t`;
oversized stride, dilation, padding, tensor, arena, or scratch values are
compile errors rather than target-side truncation or overflow.

| Tensor | Canonical layout | Shape |
|---|---|---|
| Image activation | NHWC | `[1, H, W, C]` |
| Matrix activation | NC | `[1, C]` |
| Conv weight | OHWI | `[C_out, K_h, K_w, C_in]` |
| Depthwise weight | HWO | `[K_h, K_w, C_out]` |
| Linear weight | OI | `[C_out, C_in]` |
| Bias | C | `[C_out]` |

The PyTorch frontend converts NCHW/OIHW constants to this canonical form on the
host. The generated image input ABI is NHWC and the manifest/header state that
explicitly. P0 does not emit an implicit runtime NCHW-to-NHWC copy.

Depthwise output channel `oc` reads input channel
`oc // depth_multiplier`. `C_out` must equal `C_in * depth_multiplier`.

## Quantization contract

All deployment arithmetic uses the versioned profile `bakenn.int8.v1`.

```text
activation:
    int8 [-128, 127]
    real = scale * (q - zero_point)
    per-tensor affine

weight:
    int8 [-127, 127]
    zero_point = 0
    per-output-channel symmetric

bias:
    int32
    bias_scale[c] = input_scale * weight_scale[c]

accumulator:
    int32
    wrap and saturation are forbidden

requantization:
    Q31 multiplier + signed base-two shift
    TFLM-compatible double-round profile
```

The compiler proves an accumulator bound per output channel:

```text
max_abs_x = max(abs(-128 - input_zp), abs(127 - input_zp))

bound[c] = abs(bias[c])
         + max_abs_x * sum(abs(weight[c, ...]))

bound[c] <= INT32_MAX
```

The proof also makes every MAC prefix safe for the generated int32 loop.
Positive requantization left shifts receive a second compile-time bound proof.
Scale conversion, multiplier/shift derivation, LUT generation, and bias
quantization happen once on the host; the target never derives integer
parameters from floating-point scales.

Every deployment scale is normalized at IR construction to a finite positive
IEEE-754 float32 value. Python I/O helpers, the manifest, and generated public
header use those exact bits; host-only binary64 scale drift is not permitted.

### PTQ policy

- Activations use deterministic min/max observers in P0.
- Weights use symmetric per-output-channel min/max quantization.
- ReLU-aware observations include real zero and exclude negative output range.
- Calibration requires at least one finite sample and runs the original model
  in FP32 eval semantics.
- Every graph edge owns qparams; module definition order never propagates scale.
- An all-zero weight channel with nonzero bias selects the explicit
  `output_scale / input_scale` weight scale, stores the directly quantized
  centered output code as bias, and is accepted only after exact v1 Q31 replay
  proves the generated result. The kernel may retain the zero MAC channel; P0
  does not falsely claim it was removed.
- A later percentile/entropy observer may improve accuracy without changing IR.

QAT is intentionally post-P0. It must freeze into exactly the same
`QuantizedGraph`; it cannot introduce a second deployment arithmetic path.

## Operator contracts

### Conv2D

```text
input:   NHWC int8, per-tensor
weight:  OHWI int8, per-axis axis=0, zp=0
bias:    C int32, per-axis axis=0
output:  NHWC int8, per-tensor
groups:  exactly 1
stride:  positive (h, w)
dilation: positive (h, w)
padding: explicit nonnegative (top, bottom, left, right)
```

Output shape is verified with the effective kernel
`dilation * (kernel - 1) + 1`. Padding reads `input_zero_point`, not integer
zero. Bias is added before requantization. An optional integer clamp represents
fused ReLU/ReLU6.

### DepthwiseConv2D

The activation contract matches Conv2D. Weight is HWO and per-axis on axis 2.
`C_out = C_in * depth_multiplier`; arbitrary positive depth multiplier is
supported by the portable backend. Padding also uses `input_zero_point`.

### FullyConnected

The existing NC/OI contract remains normative. Flatten-to-Linear is a view and
must not create an activation copy.

### Add

Inputs may have different affine qparams. P0 uses the TFLM/CMSIS-compatible
two-input scaling scheme:

```text
left_shift = 20
twice_max_scale = 2 * max(scale_a, scale_b)
input_a_multiplier = scale_a / twice_max_scale
input_b_multiplier = scale_b / twice_max_scale
output_multiplier = twice_max_scale / ((1 << left_shift) * output_scale)
```

Both centered inputs are shifted/scaled into a common int32 domain, added, then
requantized once to the output domain. Every intermediate has an explicit bound
proof. Broadcasting is not supported in P0: both inputs and output have exactly
the same shape. A fused activation clamp is allowed.

### Mul

Inputs and output have identical shape; broadcasting is rejected.

```text
acc = (a - zp_a) * (b - zp_b)
real_multiplier = scale_a * scale_b / output_scale
output = clamp(requantize(acc) + output_zp)
```

The product and any positive left shift must be proven int32-safe.

### ReLU and ReLU6

The quantized form is `Clamp(input, output, qmin, qmax)`. The frontend derives
the thresholds by quantizing real 0 and real 6 with the output qparams. If input
and output qparams differ, legalization inserts Requantize. The fusion pass may
move the clamp into Conv, Depthwise, Linear, Add, Mul, or Pool only when the
graph has a single legal producer/consumer path.

### AveragePool2D and MaxPool2D

Input and output must have identical qparams. Kernel and stride are positive;
padding is explicit and nonnegative. Pooling excludes padded coordinates from
the valid window count.

- MaxPool chooses the maximum raw int8 code.
- AveragePool sums centered int8 values in a proven-safe int32 accumulator,
  divides by the valid element count with half-away-from-zero rounding, and
  adds the common zero point.

An optional fused activation clamp is allowed.

### Reshape and Flatten

Input and output must have the same dtype, qparams, and number of elements.
They are view operations with no generated kernel. The memory planner maintains
alias groups. If a graph output is a view of an intermediate tensor, the
producer writes directly to the caller-owned output whenever possible.

### Concatenate

P0 supports two or more inputs with the same rank and layout, concatenated on a
static normalized axis. All non-concatenated dimensions must match. The legalized
form requires every input and output to have identical qparams; `Requantize`
nodes are inserted beforehand when necessary. The kernel is a deterministic
copy schedule and may not write in-place over any input.

### Requantize (internal)

The input and output shapes match. It applies one per-tensor Q31 multiplier and
shift to centered input codes and then applies output zero point and saturation.
It can be folded into a producer or consumer only when that preserves the
declared integer rounding points.

### Softmax

P0 supports rank-two NC input and softmax over the final dimension, with
`beta=1.0`. Output qparams are fixed:

```text
scale = 1 / 256
zero_point = -128
```

P0 defines `bakenn.softmax_lut.q15.v1`, an integer-only model-specific LUT
profile. The host creates a 256-entry table for input differences `[0, -255]`
using the input scale. Runtime subtracts the row maximum, looks up Q15 exponent
weights, sums them in a compile-time-proven uint32 domain, and emits
`clamp(round(weight * 256 / sum), 0, 255) - 128`. Positive division uses
round-half-up. The exact LUT bytes are constants in the plan and therefore part
of bit-exact testing. This profile is not claimed to be TFLite Softmax
bit-identical; a future TFLite importer must either translate through this
declared semantic difference or add a separate compatibility profile.

## IR design

The current Linear-only IR must be generalized before adding operators.

```python
Op = (
    LinearOp | Conv2DOp | DepthwiseConv2DOp |
    AddOp | MulOp | ClampOp |
    Pool2DOp | ReshapeOp | ConcatOp |
    RequantizeOp | SoftmaxOp
)
```

Every typed op exposes immutable `inputs` and `outputs`. The graph remains SSA:
each non-input/non-constant value has exactly one producer, operations are in
topological order, and all graph operations contribute to a graph output.

Verification has two levels:

1. Generic graph verification: definitions, topology, reachability, constant
   types, qparam structural validity, and static shapes.
2. Per-op verification: layouts, shape formulas, qparam rules, axes, parameter
   ranges, and constant requirements.

Stringly typed attributes and free-form dictionaries are forbidden in the IR.

## ExecutionPlan design

`QuantizedGraph` states mathematical tensor semantics. `ExecutionPlan` states
exact target execution semantics. Backends consume only the latter.

Each step contains:

- chosen kernel kind and stable arithmetic-profile ID;
- exact input/output/constant value IDs;
- Q31 multiplier/shift arrays or Add scaling parameters;
- fused clamp limits;
- proven accumulator/intermediate bounds;
- op-specific shape/stride/padding parameters;
- scratch requirement and alignment;
- Softmax LUT identity when applicable.

The plan never recomputes qparams at runtime and contains no framework types.

## Memory planning

Liveness is tensor-edge based and visits all operands of every op. The planner
must correctly retain both branches of Add/Mul/Concat until their shared use.

Storage classes are:

```text
INPUT      caller-owned read-only model input
OUTPUT     caller-owned model output
CONSTANT   read-only Flash data
ARENA      planned mutable activation storage
ALIAS      view of another tensor's storage
SCRATCH    backend temporary storage, reused between steps
```

The allocator uses alias groups plus half-open lifetimes. Buffers whose
lifetimes overlap may not overlap in the arena. The total caller allocation is
aligned `activation_arena + max_backend_scratch`. I/O bytes are reported
separately and are not hidden in the arena number.

P0 may perform in-place Clamp/Requantize only after an explicit alias-safety
pass proves the input has no later consumers and is not a caller-owned input.

## Portable C backend

The generated ABI remains model-specific and heap-free:

```c
#define BKNN_MODEL_ARENA_SIZE ...
#define BKNN_MODEL_ARENA_ALIGNMENT ...
#define BKNN_MODEL_INPUT_LAYOUT BKNN_LAYOUT_NHWC

void bknn_model_infer(
    uint8_t *restrict arena,
    const int8_t *restrict input,
    int8_t *restrict output);
```

Input, output, and arena must not overlap. `arena` may be `NULL` only when the
reported arena size is zero. Generated kernels use no variable-length arrays,
recursion, heap calls, implementation-defined signed shifts, or signed
overflow. Convolution MACs use int32 after the compiler proof; fixed-point
multiplication uses int64 intermediates only where required by Q31 semantics.

The initial portable backend prioritizes correctness and stable semantics. A
CMSIS-NN backend can later lower the same steps without modifying the IR.

## Pass order

Pass order is part of the compiler contract:

```text
import static FP32 graph
-> remove Dropout/Identity
-> fold BatchNorm
-> canonicalize layout to NHWC/OHWI/HWO
-> observe calibration edges
-> choose qparams and quantize constants
-> construct QuantizedGraph
-> verify
-> fuse legal activation clamps
-> insert required Requantize
-> analyze input-independent channels and deduplicate typed constants
-> verify again
-> lower fixed-point parameters and prove bounds
-> build alias groups and liveness
-> allocate activation/scratch arena
-> emit ExecutionPlan
-> execute integer reference / emit C
```

No optimization pass may silently change a rounding point. A pass that removes
or combines Requantize operations must prove integer equivalence, not merely
real-number equivalence.

P0's constant-channel analysis proves values but does not remove partial
channels: doing so would require typed ChannelFill/Scatter/Concat semantics in
IR, reference execution, memory planning, and every backend. Byte-identical
constants are deduplicated only when their complete TensorType and bytes match.

## Test matrix

Every op requires:

- hand-calculated integer golden vectors;
- saturation, negative values, extreme zero points, and rounding ties;
- malformed shape/qparam/parameter rejection;
- Python reference versus generated C randomized differential tests;
- GCC and Clang at `-O0`, `-O2`, strict warnings, ASan, and UBSan;
- deterministic artifact generation;
- accumulator and positive-shift overflow rejection tests.

Graph-level fixtures:

```text
TinyCNN:
  Conv -> ReLU -> MaxPool -> Conv -> ReLU -> AvgPool
  -> Flatten -> Linear -> Softmax

Residual DS-CNN:
  Conv -> Depthwise -> Conv -> Add(skip) -> ReLU -> AvgPool -> Linear

Mobile block:
  Conv1x1 -> ReLU6 -> Depthwise3x3 -> ReLU6
  -> Conv1x1 -> Add(skip) -> GlobalAveragePool -> Linear
```

For each fixture, test minimum/maximum codes, calibration samples, at least 256
seeded random integer inputs, arena canaries, and a dequantized accuracy smoke
against the FP32 model. Bit-exactness is required between integer reference and
C; FP32 comparison uses an explicit error/accuracy threshold rather than byte
equality.

## Explicit P0 exclusions

- Dynamic shapes or batch size other than one
- Runtime NCHW layout conversion
- General grouped Conv2D
- Broadcast Add/Mul
- ConvTranspose, LSTM/SVDF, resize, detection postprocess
- INT4, mixed precision, sparse weights
- QAT
- TFLite/ONNX importers
- Target-specific assembly or accelerator backends
- On-device model replacement or training

These exclusions must produce compile-time diagnostics rather than fallbacks.
