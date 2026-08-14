# BakeNN

BakeNN is a new, independent INT8 AOT compiler core for fixed-model MCU
products. It converts an already-trained FP32 model plus representative
calibration data into a model-specialized, heap-free standalone C11 library.

This directory does not depend on the legacy power-of-two implementation or on
TFLite. The current host-tested surface supports static batch-one classifiers,
depthwise/residual/SE blocks, and fixed-resolution encoder-decoder graphs.
Unsupported semantics fail during host compilation; there is no deployment-time
float fallback.

The implementation baseline compiles MobileNetV3, MobileNetV2, EfficientNet-
Lite-style, residual and compact U-Net graphs end to end. A physical
nRF52840DK comparison is checked in for a fixed FC graph and a standalone
Conv2D graph. Those measurements demonstrate the generated-C path on one
Cortex-M4 target; they are not a claim that every BakeNN model or every MCU is
faster than TFLM. See the [physical FC result](benchmarks/tflm_compare/results/iotlab_447626_direct_cmsis_fc.md)
and the [benchmark protocol](benchmarks/tflm_compare/README.md).

### First physical evidence

The following measurements are from the same frozen `32 -> 16 -> 4` INT8
fully-connected workload on an IoT-LAB nRF52840DK (Cortex-M4, 64 MHz). They
are evidence for this workload only, not a universal model or MCU ranking.

| Build | Median cycles | Flash (text+data) | SRAM (data+bss) |
|---|---:|---:|---:|
| BakeNN direct CMSIS-NN FC | 3,786 | 20,920 B | 8,540 B |
| TFLM + CMSIS-NN FC | 5,418 | 69,640 B | 11,040 B |
| BakeNN portable FC | 8,706 | 20,764 B | 8,540 B |
| TFLM reference FC | 9,342 | 63,176 B | 11,008 B |

The [standalone Conv2D report](benchmarks/tflm_compare/results/iotlab_447609_conv.md)
uses BakeNN's portable kernel versus TFLM reference and is not a CMSIS-NN
convolution comparison.

```text
PyTorch FP32 eval model + calibration samples
        -> torch.export FloatGraph
        -> PTQ QuantizedGraph + legalize/fuse
        -> verified static ExecutionPlan
        -> Python integer reference
        -> model.h + model.c + weights + portable C kernels
```

## Current numerical contract

- Activations: per-tensor affine `int8`, range `[-128, 127]`
- Weights: per-output-channel symmetric `int8`, range `[-127, 127]`
- Bias and accumulator: `int32`, with compile-time overflow proof
- Requantization: versioned Q31 double-round profile (`bakenn.int8.v1`)
- Exact underflow: ratios too small to change any int32 accumulator are emitted
  as the constant Q31 result `(multiplier=0, shift=0)`
- Deployment scales: normalized once to finite positive IEEE-754 float32
- Static shapes and batch size one only
- Unsupported operations and unsafe accumulators fail at compile time

The currently implemented operation families are:

- Conv2D (including groups), depthwise Conv2D, Conv1D (including groups), and
  FullyConnected
- statically broadcast Add/Mul, Clamp/ReLU/ReLU6, Sigmoid, HardSigmoid,
  HardSwish, SiLU, and internal Requantize
- AveragePool2D/MaxPool2D and AveragePool1D/MaxPool1D
- explicit zero Pad2D and spatial/time ReduceMean
- Reshape, Flatten, Squeeze/Unsqueeze views, static Slice/Crop, and Concatenate
- nearest and Q15 bilinear Resize2D, plus grouped ConvTranspose2D
- final-axis rank-two Softmax using `bakenn.softmax_lut.q15.v1`

The image ABI is canonical NHWC and the sequence ABI is canonical NLC. The
PyTorch frontend accepts NCHW/NCL and converts weights and activations on the
host. Pad currently supports constant real zero only; ReduceMean supports
NHWC spatial axes or the NLC time axis with either `keepdim=True` or a reduced
NC output. Sigmoid has the
fixed output qparams `(scale=1/256, zero_point=-128)`. All shapes remain static,
batch size remains one, and there is still one model input and one model output.
Resize output sizes are fixed during compilation. Bilinear resize uses the
versioned `bakenn.int8.resize_bilinear.q15.v1` coordinate/rounding profile;
ConvTranspose2D supports positive groups, static stride, dilation, asymmetric
padding and output padding with per-output-channel weights. Slice/Crop bounds,
negative indices and positive steps are normalized on the host. Add/Mul broadcast dimensions must be
statically compatible; there is no runtime broadcasting decision.

Model-level host gates currently cover:

- unmodified torchvision `mobilenet_v3_small` and `mobilenet_v3_large` graphs
  through FP32 capture, PTQ, planning and C artifact generation;
- unmodified torchvision `mobilenet_v2` at static 32x32 through FP32 capture,
  PTQ, planning, generated-C compilation and raw INT8 byte-exact execution;
- an EfficientNet-Lite-style ReLU6 MBConv classifier, plus torchvision
  EfficientNet-B0 as a harder SiLU/SE-broadcast frontend superset;
- unmodified torchvision MNASNet0.5 through FP32 capture, PTQ, planning and C
  artifact generation, including `keepdim=False` global ReduceMean;
- compact U-Nets using grouped ConvTranspose2D, nearest/bilinear resize or
  static center crop, with skip concatenation and static arena planning.
- compact ResNet bottleneck, DenseNet-style dense concat, Inception branch,
  SqueezeNet Fire, Conv1D-flatten classifier, temporal residual and Softmax
  MLP models through dequantized-accuracy and byte-exact generated-C tests.

The compact models are compiled with GCC and Clang under ASan/UBSan and their C
outputs are byte-exact with the Python INT8 reference. The real-data training
matrix covers two MNIST and four CIFAR-10 models after one epoch; its accuracy
and memory results are recorded in
[`examples/training_matrix/RESULTS.md`](examples/training_matrix/RESULTS.md).
This host matrix is separate from the physical nRF52840 evidence described
above.

The one-call PyTorch PTQ path is:

```python
import bakenn

model.eval()
compiled = bakenn.compile_torch_ptq(
    model,
    example_input,       # one static batch-one FP32 tensor
    calibration_data,    # representative FP32 tensors or batches
    "build/classifier",
    name="classifier",
)

input_q = bakenn.quantize_input(compiled.plan, input_fp32_nhwc)
output_q = bakenn.run_reference(compiled.plan, input_q)
```

The same pipeline is available as explicit inspectable stages:

```python
from bakenn.frontends import capture_torch_export

float_graph = capture_torch_export(model, example_input)
qgraph = bakenn.quantize_float_graph(float_graph, calibration_data)
compiled = bakenn.compile(qgraph, "build/classifier")
```

P2 kernel selection is explicit and reproducible. Portable C remains the
default. To allow verified shape-specialized kernels and weight packing:

```python
compiled = bakenn.compile(
    qgraph,
    "build/classifier",
    backend_options=bakenn.CBackendOptions(
        kernel_policy=bakenn.KernelPolicy.AUTO,
    ),
)
```

The generic first slice includes `optimized.linear_oi2.v1` and its odd-output
`optimized.linear_oi2_tail.v1` variant, plus `optimized.conv2d_1x1_o2.v1` and
`optimized.depthwise_3x3_c2.v1`. Unsupported shapes fall back to portable C;
`REQUIRE_OPTIMIZED` is available for coverage audits that should fail instead
of falling back. The artifact manifest records every selection, rejection
reason and packed layout. The generic candidates remain host-verified
specializations; the first measured target result is recorded separately
below.

The `cortex-m4` profile additionally has real Arm DSP-intrinsic candidates:

- `cortex_m4.linear_smlad.v1`
- `cortex_m4.conv2d_1x1_smlad.v1`
- `cortex_m4.depthwise_3x3_smlad.v1`
- `cortex_m4.conv2d_3x3_im2col_smlad.v1`
- `cortex_m4.global_average_pool2d_s8.v1`
- `cortex_m4.max_pool2d_2x2_s2.v1`

The 3x3 Conv candidate uses a reusable one-output-pixel im2col scratch region.
The Arm cross-ELF tests verify that GNU Arm emits actual `smlad` instructions
and that the final ELF has no unresolved, heap, or floating-runtime symbols.
Packed SMLAD weights can increase Flash. Target eligibility is not itself a
performance claim; the measured nRF52840 results cover only the checked-in
FC/Conv workloads.

The optional direct CMSIS-NN source backend currently covers FullyConnected on
an ARMv7E-M DSP target. It bundles only the required CMSIS-NN v4 source closure
and selects `cmsis_nn.linear_s8.v4.0.0` when the Linear has the required
per-tensor requantization contract. Conv2D and DepthwiseConv2D continue to use
BakeNN's own Cortex-M4 or portable kernels until their direct CMSIS-NN
adapters are added. Unsupported cases fall back or fail under
`REQUIRE_OPTIMIZED`.

Target selection is optional. `portable32` remains the default; ARM/RISC-V
profiles add exact ABI/alignment/compiler metadata and ESP profiles can emit an
ESP-IDF component/project:

```python
compiled = bakenn.compile(graph, "build/m4", target="cortex-m4")
report = bakenn.build_freestanding_elf(
    compiled.artifacts, "cortex-m4", "build/m4/cross"
)

esp = bakenn.compile(graph, "build/s3", target="esp32s3")
project = bakenn.export_esp_idf_project(
    esp.artifacts, "esp32s3", "build/s3/project"
)
```

The built-in target ids are `portable32`, `cortex-m0plus`, `cortex-m4`,
`rv32imc`, `esp32`, `esp32s3`, and `esp32c3`. ARM M0+/M4 and RV32IMC have a
freestanding ELF/link-map/symbol-audit path. ESP-IDF packaging and boardless
build CI are provided. Physical cycle evidence currently covers the
nRF52840DK/Cortex-M4 benchmark in
[`benchmarks/tflm_compare/results/iotlab_447626_direct_cmsis_fc.md`](benchmarks/tflm_compare/results/iotlab_447626_direct_cmsis_fc.md);
no ESP or whole-firmware energy result is claimed. See [the target-layer
contract](docs/TARGETS.md).

`AUTO` currently means "select an applicable verified specialization". It is
not a target cost model and does not promise lower latency, Flash, or energy.
Portable therefore remains the default until a physical-target measured cost
table is available. Host smoke comparisons for each implemented family
can be run with:

```bash
PYTHONPATH=src python benchmarks/host_linear_compare.py --kernel linear
PYTHONPATH=src python benchmarks/host_linear_compare.py --kernel conv1x1
PYTHONPATH=src python benchmarks/host_linear_compare.py --kernel depthwise3x3
```

Host results are regression evidence. The separate IoT-LAB result is the only
current MCU/TFLM performance evidence and is explicitly limited to its frozen
model, board, compiler and protocol.

The framework frontend captures through the real `torch.export` API and imports
PyTorch only when used. It produces immutable BakeNN-owned types immediately;
the IR, planner, reference executor, and backend do not import PyTorch. The
generated firmware never depends on PyTorch, TensorFlow, FlatBuffers, an
interpreter, C++, or dynamic allocation.

Calibration accepts arrays, tensors, iterables, and single-input
`TensorDataset`/`DataLoader` batches. Samples are snapshotted while streaming so
loaders may safely reuse backing buffers. Multi-field `(input, target)` batches
are rejected as ambiguous; pass only model inputs. Complex, object, string,
boolean, empty, and non-finite calibration data fail closed.

Eval BatchNorm is folded on the host. Dropout and Identity are removed, and
common safe in-place ReLU/Add surfaces are normalized to immutable SSA only
after mutation/fan-out checks. All-zero weight channels with nonzero bias use a
declared constant-channel scale policy whose exact INT8 output is proven under
the same Q31 arithmetic used by generated C.

Run the dependency-light test suite with:

```bash
PYTHONPATH=src python -m pytest -q
```

The end-to-end test generates C, compiles it with the host C compiler, runs it,
and compares its outputs byte-for-byte with the independent Python integer
reference.

Generated models expose a raw caller-owned arena pointer. Allocate exactly the
reported `*_ARENA_SIZE` bytes with `*_ARENA_ALIGNMENT`; pass `NULL` when the
reported size is zero. Input, output, and arena memory must not overlap. Input
and output scale/zero-point macros are emitted in the public model header.
The header also emits I/O rank, dimensions, byte counts, and canonical layout;
its `restrict` ABI requires input, output, and arena to be disjoint.

The offline comparison contract under
[`benchmarks/tflm_compare`](benchmarks/tflm_compare/README.md) records final
ELF Flash, full peak SRAM, initialization/inference cycles, and output error.
The checked-in template remains explicitly unmeasured, while the measured FC
and Conv reports are linked from that directory.

**Bake neural networks into firmware.**

The user-facing Python package is `bakenn`. Generated firmware symbols use the
short `bknn_` prefix and `BKNN_` macros; versioned numerical profiles retain
the readable `bakenn.*` namespace.

See [the P0 blueprint](docs/P0_BLUEPRINT.md) for the exact supported contract
and [the P2 kernel architecture](docs/P2_KERNEL_ARCHITECTURE.md) for selection,
fixed-point and packing contracts. The ordered post-baseline work and its
acceptance gates are tracked in [the roadmap](docs/ROADMAP.md).
