# Trained MNIST evidence

This directory is produced by `examples/mnist/run_mnist.py` and freezes the
inputs needed to audit or replay the trained demonstration:

- the serialized FP32 state-dict checkpoint and a serialization-independent
  logical tensor hash;
- the exact class-balanced calibration images and labels;
- a class-balanced physical-board input corpus, its quantized INT8 bytes,
  labels and Python integer-reference outputs;
- standalone generated C11 artifacts and their deterministic set hash;
- the host FP32/generated-C accuracy report.

`mnist_evidence.json` is the root manifest. Every included payload has a byte
count and SHA-256. The calibration and physical corpus hashes also commit to
dtype, shape and labels, so raw files cannot be reordered without detection.

Replay the frozen checkpoint without retraining:

```bash
PYTHONPATH=src python examples/mnist/run_mnist.py \
  --checkpoint examples/mnist/evidence/mnist_fp32.pt
```

Generate the self-contained original-ESP32 project from the same checkpoint,
calibration corpus and expected output bytes:

```bash
PYTHONPATH=src python examples/mnist/generate_esp32_benchmark.py
```

Host execution and boardless ESP-IDF compilation are correctness/toolchain
evidence. Only a checked-in UART transcript from a physical board may be used
as latency evidence.
