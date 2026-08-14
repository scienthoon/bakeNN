# Recorded MNIST run

Recorded on 2026-08-14 with BakeNN 0.1.0, PyTorch 2.10.0, the default seed
`20260814`, four training epochs, and 160 class-balanced calibration images.

```text
Training images:                 60,000
Test images:                     10,000
FP32 test accuracy:               96.92%
Generated-C INT8 test accuracy:   97.02%
Accuracy difference:              +0.10 percentage points
Python/C compared output bytes:    1,280
Python/C mismatched bytes:              0
Activation arena:                 3,920 bytes
Emitted constants:                4,508 bytes
```

The generated C was executed on all 10,000 test inputs. Byte-exact comparison
against the Python integer reference used the first 128 inputs, producing ten
output bytes per input. The host execution time in `mnist_report.json` includes
process pipes and is not an MCU performance claim.

The generated ABI for this run was:

```c
#define BKNN_MNIST_ARENA_SIZE 3920u

void bknn_mnist_infer(
    uint8_t *restrict arena,
    const int8_t *restrict input,
    int8_t *restrict output);
```
