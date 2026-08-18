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
Physical corpus reference:       99/100 correct
```

The generated C was executed on all 10,000 test inputs. Byte-exact comparison
against the Python integer reference used the first 128 inputs, producing ten
output bytes per input. The host execution time in `mnist_report.json` includes
process pipes and is not an MCU performance claim.

The frozen evidence identities are:

```text
checkpoint file SHA-256:   a2f08c02a718d2e6cbe590ad442891db04308a3c82a2c791b8e0fc02186ea498
checkpoint logical hash:   733130916e926da785afc63da0786d7613b0db808c3482398ae84b41ab706840
calibration corpus hash:   06ce2a1aaaa0f75e566b9494c17b3c1fce4063272d1ce4e879c24b868bbb4e29
generated artifact set:    cc59172a18cc8d752185ab76c4a5a50c1bdccbcbf9abf8d8457a013e63d28f12
physical INT8 input hash:  27c96cc7e1e5b49058b8183781af349c1773b89cbfa4791b6eb098397111f653
expected output hash:      cdbbfc38bf5f17a2867b17a8a28e643043e50733063e246b0bc85cbeaae30ba4
```

These values are also machine-readable in
[`evidence/mnist_evidence.json`](evidence/mnist_evidence.json). The ESP32
cross-build is not presented as a physical benchmark until its UART transcript
is checked in.

The generated ABI for this run was:

```c
#define BKNN_MNIST_ARENA_SIZE 3920u

void bknn_mnist_infer(
    uint8_t *restrict arena,
    const int8_t *restrict input,
    int8_t *restrict output);
```
