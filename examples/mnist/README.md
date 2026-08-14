# BakeNN MNIST end-to-end example

This example trains a compact supported PyTorch CNN, calibrates it with a
class-balanced subset, generates standalone C11 through BakeNN, and evaluates
the generated C on all 10,000 MNIST test images. It also requires exact output
bytes between the Python integer reference and generated C on a fixed subset.

From the repository root:

```bash
PYTHONPATH=src python examples/mnist/run_mnist.py
```

The dataset is cached under `/private/tmp/bakenn-mnist-data` by default. Build
artifacts, the trained state dict, and `mnist_report.json` are written under
`examples/mnist/build`. Use `--help` to change epochs, calibration size,
compiler, or output directories.

Host execution time is informational and is not an MCU performance benchmark.
The meaningful accuracy fields are FP32 test accuracy, generated-C test
accuracy, and the Python/C mismatched-byte count, which must remain zero.

See [`RESULTS.md`](RESULTS.md) for the checked-in result of the default run.
