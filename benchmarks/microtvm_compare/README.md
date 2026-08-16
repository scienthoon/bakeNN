# microTVM AOT + USMP + CMSIS-NN comparison

This benchmark compares two full-model Cortex-M4 build paths under one frozen
quantized contract:

1. BakeNN lowers the trained MNIST graph directly to CMSIS-NN calls.
2. The same `QuantizedGraph` is serialized as a fully quantized `.tflite`,
   imported by Apache TVM 0.16.0, partitioned to CMSIS-NN, statically planned
   by USMP and emitted with the AOT C interface.

This is a stronger comparison than feeding separately calibrated models to the
two compilers. The FP32 checkpoint, 160 calibration images, activation qparams,
INT8 weights, input bytes and expected output bytes are shared. The comparison
uses per-tensor FullyConnected weights because TVM 0.16's CMSIS-NN importer
requires a scalar FC weight scale.

## Checked-in result

[`results/mnist_cortex_m4_cross_build.json`](results/mnist_cortex_m4_cross_build.json)
records:

- 100 class-balanced trained-MNIST samples, 99 correct;
- 1,000 output bytes compared between the BakeNN integer reference and the
  microTVM-generated CMSIS-NN C path, with **0 mismatches and 0 LSB maximum
  error**;
- five CMSIS-NN partitions covering two convolutions, two max pools and one
  fully connected layer;
- USMP workspace of 4,064 bytes;
- successful freestanding Cortex-M4 links with no unresolved symbols;
- 12,088 bytes linked Flash for BakeNN versus 17,456 bytes for microTVM in
  this exact cross-build.

The last numbers are **linker evidence, not a latency measurement**. They prove
what each compiler placed in this Cortex-M4 ELF. Only a same-board run may be
used to compare cycles, stack high-water or energy.

## Reproduce

The baseline is the official Apache TVM 0.16.0 source release. Its source
archive SHA-512 is:

```text
e2d7f81ed87d184fdd20b7e1f2fd16bf7e15a52aea9c52fde95cb1444101e64588a8a1d0d360f3cd60d72cad01d619195ce45eefec7d07b6544888da8252609b
```

Build TVM with microTVM and CMSIS-NN enabled:

```bash
curl -LO https://downloads.apache.org/tvm/tvm-v0.16.0/apache-tvm-src-v0.16.0.tar.gz
tar -xf apache-tvm-src-v0.16.0.tar.gz
cmake -S apache-tvm-src-v0.16.0 -B tvm-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_MICRO=ON \
  -DUSE_CMSISNN=ON \
  -DUSE_LLVM=OFF
cmake --build tvm-build --parallel
```

Install BakeNN's PyTorch extra plus the TVM Python and TFLite schema
dependencies in an isolated environment. Then run:

```bash
PYTHONPATH=apache-tvm-src-v0.16.0/python:src:examples/mnist:benchmarks/tflm_compare \
TVM_LIBRARY_PATH="$PWD/tvm-build" \
TOPHUB_LOCATION=NONE \
python benchmarks/microtvm_compare/build_mnist.py \
  --tvm-source apache-tvm-src-v0.16.0 \
  --tvm-build tvm-build
```

Required host tools are a C/C++ compiler and `arm-none-eabi-gcc`. The script
fails unless:

- all frozen checkpoint and calibration hashes match;
- the expected TFLite operator set is produced;
- all five operations are partitioned to CMSIS-NN;
- the generated C contains the expected `arm_*_s8` calls;
- 1,000 output bytes match the BakeNN reference exactly;
- both freestanding Cortex-M4 ELFs link without unresolved symbols.

TVM 0.16 uses an LLVM host function to evaluate a compile-time weight
transpose even when the target is the C backend. To keep the build independent
of a version-matched LLVM, the script evaluates that constant-only function
with TVM's C backend and the host C++ compiler. This does not change the target
Relay graph, AOT executor, USMP plan or CMSIS-NN code path, and the mechanism is
recorded in the result JSON.

## Evidence files

The results directory includes the common `.tflite`, expected INT8 outputs,
TVM Relay, unmodified MLF-generated model sources/header/metadata, and BakeNN
manifest/memory report. Every file hash and the combined evidence-set hash are
recorded in the result JSON.

Official references:

- [Apache TVM 0.16 microTVM custom IDE tutorial](https://tvm.apache.org/docs/v0.16.0/how_to/work_with_microtvm/micro_custom_ide.html)
- [Apache TVM Model Library Format](https://tvm.apache.org/docs/v0.16.0/arch/model_library_format.html)
- [Apache TVM 0.16 source archive](https://downloads.apache.org/tvm/tvm-v0.16.0/)

