# BakeNN P2 kernel backend architecture

Status: generic optimized slice plus cross-compiled Cortex-M4 DSP kernels; no
physical-MCU performance claim

P2 adds backend decisions without changing the mathematical model or the
static memory plan. The boundary is deliberately strict:

```text
QuantizedGraph
    -> ExecutionPlan          semantic ops, Q31 parameters, bounds, arena
    -> CBackendPlan           selected implementation and packed constants
    -> C11 artifacts          calls, kernels, weights, decision manifest
```

`ExecutionPlan` remains the source of truth for Python integer reference
execution and numerical correctness. `CBackendPlan` is a read-only overlay: it
may choose an equivalent implementation and a backend-owned representation of
a constant, but it may not change tensor qparams, rounding, activation bounds,
liveness, activation offsets, or operation order. It may enlarge the final
arena with one reusable backend scratch region: the maximum selected-kernel
size and alignment are placed after the activation arena and reported in the
generated ABI and manifest.

## Public interface

The generated-C backend accepts `CBackendOptions`:

```python
options = bakenn.CBackendOptions(
    kernel_policy=bakenn.KernelPolicy.AUTO,
    enable_weight_packing=True,
)
compiled = bakenn.compile(graph, "build/model", backend_options=options)
```

The policies are:

- `PORTABLE`: select only the portable baseline. This is the default.
- `AUTO`: choose the highest-priority supported candidate, then use the stable
  implementation ID as the deterministic tie-breaker.
- `REQUIRE_OPTIMIZED`: reject compilation if a step has no applicable
  optimized implementation.

Every family registers capability records through a typed dispatch hook. A
record includes a versioned implementation ID, priority, applicability result,
exact reason, optimized/baseline classification, packed constants, and semantic
constant overrides plus scratch size/alignment. Selection is performed once on the host. The selected ID,
reason, rejected candidates and representation metadata are written to the
manifest.

Unsupported shapes never enter a specialized kernel. `AUTO` falls back to the
portable implementation; `REQUIRE_OPTIMIZED` fails closed.

## Fixed-point contract

All integer backends implement the versioned `bakenn.int8.v1` contract already
stored in the graph and plan:

- activation codes are affine signed INT8: `real = scale * (q - zero_point)`;
- weights are symmetric per-output-channel INT8 with zero point zero;
- bias and MAC accumulation are INT32;
- every accumulator and positive pre-high-multiply shift is proven INT32-safe
  on the host before C generation;
- the host encodes a positive real multiplier as normalized Q31 multiplier plus
  a shift in `[-31, 30]`;
- requantization performs saturating rounding doubling high multiply, followed
  by round-half-away-from-zero division by a power of two;
- output zero point is added in INT64, then clamped to the fused activation
  interval and INT8 storage range;
- signed overflow and signed negative shifts are not target-language semantics.

The generated model now contains one common C implementation of this Q31 path.
Conv, DepthwiseConv, Linear, Add, Mul and Requantize call that implementation;
families do not carry independent copies of the rounding formula. Python
reference, portable C and an optimized C kernel must be byte-exact for every
accepted input. An implementation that uses approximate SIMD rounding needs a
new arithmetic-profile ID or must be rejected; it cannot silently claim v1.

## Packing rules

Packing is a lowering concern, never an IR mutation:

1. The source must be a verified immutable semantic constant in the
   `ExecutionPlan`.
2. A packed value is a private immutable C-contiguous INT8 or INT32 snapshot.
3. Every representation has a versioned layout string and power-of-two
   alignment requirement.
4. An override may replace only a constant consumed by that exact step.
5. Packed names cannot collide with semantic constant names. Reused names must
   have identical source, layout, alignment, dtype, shape and bytes.
6. An override may point only to a packed representation whose declared source
   is that exact semantic constant; weight/bias cross-wiring is rejected.
7. The original constant is omitted from the C artifact only when no selected
   step still consumes it; mixed consumers retain both representations.
8. Packing must not change channel order, per-channel multiplier indexing, MAC
   order within a channel, qparams, or numerical results.
9. Packed payloads, alignment, backend scratch offsets and the final arena must
   all fit the declared 32-bit target ABI.

These rules make packing deterministic and auditable while leaving the Python
reference independent of backend layout.

## First optimized kernels: Linear, 1x1 Conv and Depthwise 3x3

The first slice is intentionally small. Linear uses two versioned variants:

- `optimized.linear_oi2.v1` for even output counts;
- `optimized.linear_oi2_tail.v1` for an odd output count, with one unpacked
  scalar tail row appended after the packed output pairs.

The pair variant is applicable when:

- the verified operation is Linear with OI INT8 weights;
- output features are at least two and divisible by two;
- the MAC count is at least 48;
- weight packing is enabled.

The host packs each pair of output rows as input-major pairs. The kernel keeps
two INT32 accumulators live and reuses each centered input value for both output
channels. Accumulation order within each output channel is unchanged, so v1
overflow proofs and byte-exact results remain valid. Small shapes, odd shapes
without enough work, and disabled packing use the portable OI kernel under
`AUTO`.

The first convolution candidates are:

- `optimized.conv2d_1x1_o2.v1`: zero-padded-free 1x1 OHWI Conv2D, stride
  components 1 or 2, even output channels, output-pair packing;
- `optimized.depthwise_3x3_c2.v1`: HWO 3x3 DepthwiseConv2D, depth multiplier
  one, dilation one, stride components 1 or 2, even channel-pair packing, and
  explicit asymmetric padding.

Both preserve the existing input-zero-point padding semantics and fall back to
portable C for odd channels, unsupported stride/dilation, non-matching kernel
shapes, or disabled packing.

These generic kernels remain reference optimized implementations and an
architecture proof. Host speed is not an MCU speed claim.

## Cortex-M4 DSP kernels

The `cortex-m4` target adds a separate versioned family that requires both the
`armv7e-m` and `dsp` target features:

- `cortex_m4.linear_smlad.v1`;
- `cortex_m4.conv2d_1x1_smlad.v1`;
- `cortex_m4.depthwise_3x3_smlad.v1`;
- `cortex_m4.conv2d_3x3_im2col_smlad.v1`;
- `cortex_m4.global_average_pool2d_s8.v1`;
- `cortex_m4.max_pool2d_2x2_s2.v1`.

The MAC kernels pack two signed INT8 weights as two sign-extended INT16 lanes
and invoke `__builtin_arm_smlad`. The general 3x3 kernel lowers one output
pixel at a time into a statically planned scratch patch, then reuses the same
Q31 helpers as every other backend. Unsupported groups, shapes, strides,
dilations or targets take the declared `AUTO` fallback or exact
`REQUIRE_OPTIMIZED` error. Grouped Conv2D remains available through portable C;
the Cortex-M4 convolution candidates currently require `groups=1` except for
the dedicated depthwise case.

Cross-toolchain tests link a freestanding Cortex-M4 ELF, disassemble it to
confirm actual `smlad` instructions, and audit undefined/heap/soft-float
symbols. This proves target code generation, not that the kernels are faster.
The lane-expanded packed representation can cost more Flash than semantic INT8
weights, and the built-in measured cost table remains empty until board runs
exist.

## ESP-NN target overlay

ESP-NN is an explicit, opt-in target overlay; it does not change the portable
graph or fixed-point contract. BakeNN pins ESP-NN 1.2.6 at revision
`c0876179f1cf4b4b9073b4f81cb65c8051ccb476`, bundles the corresponding source
closure into the generated ESP-IDF component, and records the dependency and
requantization configuration in the manifest.

The ESP32-S3 family covers Conv2D, DepthwiseConv2D, per-channel Linear,
AveragePool2D and MaxPool2D. Conv/Depthwise global scratch and the guarded,
aligned Linear input staging area are represented by `CBackendPlan.scratch`
and overlaid with the activation arena. The original ESP32 uses ESP-NN's
optimized Conv/Depthwise C path; its ESP-NN FC/pool entry points are ANSI
implementations, so selection retains BakeNN's existing generic kernels there.
ESP32-C3 is unsupported by this ESP-NN revision and falls back normally.

All predicates are semantic, not merely shape checks: they include layout,
groups/depth multiplier, dilation, padding symmetry where required, channel
alignment, zero points, per-channel multipliers/shifts, fused clamp and pooling
rounding. `CONFIG_NN_SKIP_NUDGE` is forbidden because it would change the
declared double-rounding result. Unsupported cases are portable fallback under
`AUTO` and exact compile errors under `REQUIRE_OPTIMIZED`.

The original ESP32 optimized sources execute in the host differential suite.
ESP32-S3's Xtensa assembly cannot execute on the host, so the wrappers are
checked against ESP-NN's official ANSI implementation and the assembly source
closure is cross-compiled by ESP-IDF CI. Neither proves physical-device speed;
S3 cycle, cache, peak stack and energy results remain a measurement gate.

## Acceptance and review gates

The P2 slice is accepted only if:

- portable remains the default and existing artifacts stay correct;
- selection and packing are deterministic and immutable;
- unsupported specialization is either portable fallback or an exact compile
  error according to policy;
- Python reference, portable C and optimized C match byte-for-byte on hand
  edges and at least 10,000 randomized inputs per optimized family, including
  positive/negative Q31 shifts, extreme zero-points, asymmetric padding,
  shared packed weights and an INT32 accumulator boundary;
- strict C11 builds pass at `-O0`, `-O2` and `-Os` with warnings-as-errors,
  ASan and UBSan;
- the manifest records the reproducible implementation and packed layout;
- no smaller/faster claim is made without same-target ELF, SRAM and cycle data.

The target-facing package now provides immutable target descriptors,
ARM/RISC-V freestanding ELF/map/symbol verification, ESP-IDF packaging, and the
declared Cortex-M4 DSP family. Its built-in measured cost tables are
intentionally empty. The next backend work must collect physical-target
measurements, add an evidence-backed cost model, and only then make fastest
kernel claims. It must benchmark final linked artifacts rather than infer
performance from host source structure.

## Final review disposition

No correctness blocker remains in the declared first slice. Reviews found and
fixed integration hazards around Q31 drift, consumer-aware constant retention,
scratch overlay, packed-source cross-wiring and backend memory exceeding the
32-bit target ABI. These are now executable contracts rather than family
conventions.

The following are explicit next-phase limitations, not hidden fallbacks:

- the portable target has only the declared generic Linear, 1x1 Conv2D and 3x3
  depthwise candidates; Cortex-M4 additionally has DSP Linear, 1x1, depthwise,
  general 3x3-with-scratch and specialized pool candidates; opt-in ESP32-S3
  adds ESP-NN Conv/Depthwise/Linear/pool and original ESP32 adds its optimized
  Conv/Depthwise source path;
- selection priority is a static policy; built-in target cost tables have zero
  entries until physical evidence exists;
- the generic OI2 family remains standard C; the Cortex-M4 family is a distinct
  intrinsic implementation and does not alter the generic artifact;
- the host smoke benchmark is useful only for regression detection;
- nRF52840 Flash, linked SRAM, cycles and output checksums are measured for the
  frozen FC and standalone Conv reports; full peak SRAM decomposition,
  initialization cycles, energy and ESP targets remain unmeasured.

Until same-device evidence is added, portable remains the default and BakeNN
must not claim that `AUTO` is universally faster.
