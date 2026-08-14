# One-epoch real-data model matrix

This experiment trains two MNIST and four CIFAR-10 classifiers for one complete
epoch, calibrates each trained FP32 checkpoint with 100 class-balanced training
images, generates standalone C, runs all 10,000 test images through that C, and
compares a fixed output subset byte-for-byte with the Python integer reference.

The CIFAR-10 run uses the standard FastAI image-folder mirror because the
original Toronto download endpoint can be unusually slow:

```bash
mkdir -p /private/tmp/bakenn-datasets/cifar10-fastai
curl -L -o /private/tmp/bakenn-datasets/cifar10-fastai.tgz \
  https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz
tar -xzf /private/tmp/bakenn-datasets/cifar10-fastai.tgz \
  -C /private/tmp/bakenn-datasets/cifar10-fastai --strip-components=1
```

Run the full matrix from the repository root:

```bash
PYTHONPATH=src python examples/training_matrix/run_one_epoch.py
```

Checkpoints, generated code, per-model JSON and the aggregate `RESULTS.md` are
written under `examples/training_matrix/build`. One epoch is a pipeline and PTQ
smoke test, not a competitive dataset-accuracy benchmark. Host training and C
execution timings must not be presented as MCU performance.

The checked-in result from the default six-model run is in
[`RESULTS.md`](RESULTS.md).
