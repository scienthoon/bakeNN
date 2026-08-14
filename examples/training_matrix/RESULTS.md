# Recorded one-epoch real-data matrix

Recorded on 2026-08-14 with BakeNN 0.1.0, PyTorch 2.10.0 and seed
`20260814`. Training used the complete MNIST/CIFAR-10 training splits for one
epoch on CPU. Each trained checkpoint was calibrated with 100 class-balanced
training images. Apple Clang 21 compiled the generated C, which then ran all
10,000 test images for every model.

| Model | Dataset | FP32 | INT8 C | FP32-INT8 (pp) | Train s | Arena | Constants |
|---|---|---:|---:|---:|---:|---:|---:|
| MNIST MLP | MNIST | 92.69% | 92.68% | +0.01 | 2.2 | 64 B | 51,704 B |
| MNIST CNN | MNIST | 92.53% | 92.52% | +0.01 | 10.4 | 3,920 B | 4,508 B |
| Bottleneck | CIFAR-10 | 20.20% | 20.22% | -0.02 | 50.3 | 36,864 B | 1,464 B |
| Dense concat | CIFAR-10 | 18.83% | 19.34% | -0.51 | 56.5 | 40,960 B | 1,408 B |
| Inception | CIFAR-10 | 18.64% | 18.36% | +0.28 | 52.7 | 28,672 B | 856 B |
| Fire | CIFAR-10 | 18.59% | 18.69% | -0.10 | 54.4 | 20,480 B | 848 B |

`FP32-INT8` is positive when PTQ reduced accuracy and negative when the INT8
result happened to be higher. Every row used 60,000/50,000 training examples,
10,000 test examples and 100 calibration examples. Python INT8 and generated C
were compared on 160 output bytes per model with zero mismatches in all six
runs. The repository's randomized operator/model tests provide the broader
GCC/Clang differential coverage.

The low absolute CIFAR-10 accuracy is expected from these deliberately shallow
models after only one epoch. This experiment validates the real training ->
calibration -> PTQ -> generated-C pipeline and its quantization delta; it is not
a competitive CIFAR-10 result or an MCU performance benchmark. Host totals were
226.5 seconds for training, 161.4 seconds for PTQ/code generation and 24.8
seconds for generated-C test execution including process pipes.
