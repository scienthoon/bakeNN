# BakeNN P0 status

Status: host implementation baseline complete; Cortex-M4 and original-ESP32
TFLM evidence recorded for frozen workloads

This page separates code that exists and has automated coverage from release
gates that still require integration or target evidence. Passing operator tests
does not imply that BakeNN is faster or smaller than TFLM.

## Implementation matrix

| Area | Current state | Evidence / limitation |
|---|---|---|
| Typed INT8 graph and verification | Implemented | Immutable static values/ops; rejects unsupported topology, dtype, shape and unsafe arithmetic. |
| Compute ops | Implemented and host-tested | Conv2D including groups, DepthwiseConv2D, Conv1D including groups, grouped ConvTranspose2D, and FullyConnected reference/portable-C paths. P2 also has generic specializations and a cross-compiled Cortex-M4 DSP family. |
| Elementwise and activation | Implemented and host-tested | Static-rank broadcast Add/Mul, Clamp, Sigmoid, HardSigmoid, HardSwish, SiLU and internal Requantize; Clamp fuses into eligible producers including Requantize without removing its rounding point. |
| Pool, tensor, shape and output | Implemented and host-tested | 2D/1D Avg/MaxPool, zero Pad2D, spatial/time ReduceMean with retained or removed reduced dimensions, NHWC/NLC Flatten and Reshape/Squeeze/Unsqueeze views, static Slice/Crop, Concatenate, static nearest/Q15-bilinear Resize2D, and BakeNN Q15-LUT Softmax. Softmax is explicitly not TFLite bit-exact. |
| Static memory plan | Implemented and host-tested | Multi-input liveness, reuse, view/in-place safety and shared scratch planning. Physical-target stack usage is not inferred by this planner. |
| Portable C11 emission | Implemented and host-tested | Heap-free generated artifacts, strict compiler and sanitizer coverage, Python/C integer differential tests. |
| P2 kernel selection and packing | Generic and target overlays implemented | Deterministic policy/capability overlay, immutable versioned packed constants, exact source/override checks, scratch/resource bounds, decision manifest and portable fallback. Cortex-M4 Linear/Conv kernels use real SMLAD intrinsics and cross-link cleanly. Direct pinned CMSIS-NN v4 adapters cover FC, Conv2D, DepthwiseConv2D, AveragePool2D and MaxPool2D. Pinned ESP-NN 1.2.6 covers S3 Conv/Depthwise/FC/pool and original ESP32 optimized Conv/Depthwise as an opt-in. Both vendor backends place required scratch in the static arena. |
| PyTorch capture | Implemented for the declared extended surface | Lazy `torch.export`; static batch one NCHW/NCL; grouped 2D/1D Conv and ConvTranspose2D, static Slice/Crop, HardSigmoid/HardSwish/SiLU, static broadcast, resize, pad, mean, pool and shape surfaces; eval BN2D/BN1D folding; Dropout/Identity removal; safe in-place surfaces normalized after mutation/fan-out checks. |
| PTQ | Implemented for declared graph surface | Deterministic min/max observation, per-tensor activation and per-channel weight quantization. No QAT in P0. |
| Representative quantized fixtures | Implemented | Hand-built TinyCNN, residual DS-CNN and MobileNetV1-style graphs exercise whole QuantizedGraph-to-C paths. |
| FP32 PyTorch to C for all three release fixtures | Implemented and host-tested | TinyCNN, residual DS-CNN and Mobile block pass public `compile_torch_ptq`, dequantized FP32 smoke, integer reference, and strict sanitized generated C. |
| Model-family expansion | Implemented and host-tested for the declared surface | Compact models cover MobileNetV3 SE, MobileNetV2, EfficientNet-style MBConv, U-Net decode/skip paths, ResNet bottleneck, DenseNet-style concat, Inception branches, SqueezeNet Fire, Conv1D flatten, temporal residual and Softmax MLP through sanitized generated C. Unmodified torchvision MobileNetV3-small/large, MobileNetV2, EfficientNet-B0 and MNASNet0.5 reach generated artifacts at 32x32. One-epoch MNIST/CIFAR accuracy and memory results are recorded in the training matrix. |
| Constant-channel / degenerate-range handling | Implemented and host-tested | Zero dynamic range uses scale=1/zp=0. Zero-weight/nonzero-bias channels use an explicit output-domain scale/bias policy and exact v1 replay proof; the channel is retained rather than falsely reported as folded away. |
| Generated resource manifest and memory report | Partially implemented | Emits deterministic JSON/text reports for constants, activation lifetimes, arena reuse, selected-kernel scratch, I/O and alignment. It explicitly leaves final Flash, whole-firmware SRAM and stack to target ELF/map or physical measurement. |
| Target descriptors and cross-link | Implemented and host-tested | Versioned portable32, Cortex-M0+/M4, RV32IMC and ESP profiles. ARM/RISC-V freestanding ELF/map and symbol audits pass with real GNU embedded toolchains; Cortex-M4 disassembly contains the selected SMLAD instructions. Physical nRF52840 evidence is recorded for the FC and standalone Conv benchmark only. |
| ESP-IDF packaging | Implemented; boardless CI and original-ESP32 evidence | Self-contained component/project and cycle/stack/output runner for ESP32/S3/C3. CI builds the ESP-NN Conv/Depthwise smoke on ESP32/S3 and the deterministic fallback on C3. A physical original ESP32 ran a trained MobileNetV2-0.25 through portable C, direct ESP-NN and TFLM+ESP-NN with identical output bytes. ESP32-S3 remains boardless-only. |
| TFLM comparison harness | Implemented for first frozen workloads | Offline protocol, versioned result shape, dependency-free validator, TFLM FC/Conv runners and pinned CMSIS-NN build path. Physical result reports record identical FC outputs and a separate standalone Conv comparison. |
| Physical MCU benchmark | **Partial evidence** | nRF52840DK/Cortex-M4 final ELF sizes, arena, median/p95 cycles and output checksums are measured for frozen FC and standalone Conv workloads. Original ESP32 MobileNetV2-0.25 portable/direct-ESP-NN/TFLM paths include cycles, linked resources and exact output checks. Full peak SRAM decomposition, initialization cycles, energy, ESP32-S3 and other models remain unmeasured. |
| Wheel/clean install release gate | Passed locally; CI encoded | A built 0.1.0 wheel installed into a fresh venv and compiled/tested all three PyTorch-to-C fixtures from `site-packages`; the CI workflow repeats this gate. |

## Verified properties

- Quantized operator reference implementations and emitted C are covered by
  hand goldens, malformed-input rejection, randomized differential tests, and
  strict C11/sanitizer tests in the repository.
- Compile-time checks cover graph topology, static dimensions, quantization
  parameters, accumulator bounds, positive requantization shifts, alias safety,
  scratch/activation allocation overlap, packed-source identity, and all
  backend-owned payload/arena sizes under the 32-bit target ABI.
- P2 optimized families have dedicated 10,000-input differential tests plus
  extreme zero-point, asymmetric padding, positive/negative shift, shared
  weight, mixed-kernel whole-model and near-INT32 accumulator coverage.
- Generated target code has no intended PyTorch, TFLite, FlatBuffers,
  interpreter, C++ runtime, heap-allocation, or runtime graph-preparation
  dependency.
- PyTorch semantic capture folds eval BatchNorm into new FP32 producer
  constants before PTQ. It rejects training BatchNorm and unsafe shared
  producers rather than exporting a BN target op.
- The benchmark schema requires an explicit `unmeasured` state, preventing this
  repository's template from being mistaken for performance evidence.
- The current host matrix passes the full suite with both `CC=gcc` and
  `CC=clang`; representative generated artifacts are compiled as strict C11
  with ASan/UBSan and compared byte-for-byte to the integer reference.

## Remaining release gates

1. Extend the frozen-model TFLM exporter/resolver to TinyCNN, residual and
   reduced MobileNet/CIFAR workloads if a full model-by-model TFLM table is
   required.
2. Measure full peak SRAM decomposition, initialization cycles and energy only
   when the target runner can instrument them; current reports explicitly leave
   unavailable components unmeasured.
3. Measure the new direct CMSIS-NN Conv2D/Depthwise adapters against TFLM using
   identical kernels and frozen convolution-heavy models before making a
   same-kernel performance claim.
4. Repeat the original-ESP32 measurement from a clean release tag with multiple
   input tensors, and run ESP32-S3 physically before introducing a general ESP
   target cost table or S3 acceleration claim.
5. Re-run the clean-wheel, compiler/sanitizer and public-API gates after the
   target backend and comparison integration freeze.

The compiler baseline is usable for the declared host-tested surface. The
current speed claims are intentionally limited to the frozen nRF52840 and
original-ESP32 workloads in the checked-in reports; they must not be
generalized to all models or targets.
