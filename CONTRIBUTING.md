# Contributing to BakeNN

BakeNN welcomes bug reports, documentation improvements, model-compatibility
fixtures and carefully scoped operator or kernel contributions. The project is
alpha software, so public API changes must be discussed before implementation.

## Development setup

```bash
git clone https://github.com/scienthoon/bakeNN.git
cd bakeNN
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,torch]'
pytest -q
```

The normal suite exercises GCC and Clang when available. CI additionally
cross-links Cortex-M and RV32IMC artifacts and builds ESP-IDF projects for
ESP32, ESP32-S3 and ESP32-C3.

## Before opening a pull request

- Keep the framework frontend, quantized IR, execution plan and C backend
  separated. Target emitters must consume verified plan semantics.
- Preserve `bakenn.int8.v1`; do not introduce a second rounding formula inside
  a kernel. New arithmetic profiles require explicit versioning.
- Unsupported shapes or semantics must fail closed or use an explicit `AUTO`
  fallback. Never add target-side float fallback.
- New kernels need versioned kernel/layout IDs, exact capability predicates,
  manifest reporting and byte-exact comparison with the Python reference.
- Generated C must remain C11, heap-free and clean under
  `-Wall -Wextra -Werror -pedantic`, ASan and UBSan where applicable.
- Do not claim target speed from host timing or a cross-build. Include the
  board, compiler, flags, model/input hashes, raw measurements and output
  parity for performance claims.
- Add malformed-input tests as well as happy-path tests. Do not weaken existing
  golden values or sanitizer settings.

Useful focused commands:

```bash
pytest -q tests/p0                    # IR, PTQ, reference and portable C
pytest -q tests/p2                    # optimized/vendor kernels
pytest -q tests/p3                    # expanded models and frontends
pytest -q tests/targets               # target packaging and cross-linking
python -m pip wheel . --no-deps -w wheelhouse
```

## Reporting compatibility requests

For an unsupported model, include the smallest reproducible `torch.nn.Module`,
the example input shape, the exact `CompileError`, Python/PyTorch/BakeNN
versions and whether changing the public static batch-one, single-input/output
ABI would be required. Requests for dynamic shapes or a runtime model loader
are outside the current product contract.

## Pull requests

Keep each pull request focused. Explain the numerical contract, fallback
behavior, generated-code impact and tests. By submitting a contribution, you
agree that it is licensed under the repository's Apache-2.0 license.
