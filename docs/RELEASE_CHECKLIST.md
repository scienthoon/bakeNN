# BakeNN release checklist

## Code and numerical gates

- [ ] `pytest -q` passes with GCC and Clang.
- [ ] PyTorch frontend tests pass for every supported Python/PyTorch pair.
- [ ] ARM and RISC-V freestanding cross-link tests pass with no unresolved,
      heap or floating-runtime symbols.
- [ ] ESP-IDF projects build for ESP32, ESP32-S3 and ESP32-C3.
- [ ] Generated C remains byte-exact with the Python INT8 reference.
- [ ] New benchmark claims include raw evidence and explicit scope.

## Packaging

- [ ] Version agrees in package metadata, import and generated manifest.
- [ ] Wheel and source distribution build from a clean checkout.
- [ ] A fresh environment installs the wheel and runs the smoke compiler.
- [ ] Vendored licenses and pinned revisions are present in the wheel.
- [ ] `CHANGELOG.md` replaces `unreleased` with the release date.
- [ ] PyPI project `bakenn` trusts repository `scienthoon/bakeNN`, workflow
      `release.yml`, environment `pypi` through Trusted Publishing.
- [ ] The GitHub `pypi` environment has any intended approval policy and the
      tag workflow receives only `id-token: write` for the publish job.

## Repository and release

- [ ] CI is green on the exact release commit.
- [ ] README support/limitation tables match the implementation.
- [ ] Security, contribution and conduct files are present.
- [ ] Release notes link the benchmark evidence without broadening its claims.
- [ ] Create and push an annotated `vX.Y.Z` tag from `main`.
- [ ] Publish GitHub release artifacts and verify their SHA-256 hashes.
- [ ] Attach the deterministic evidence ZIP containing JSON/CSV, raw UART and
      the exact retained ELF/map files; document any historical dirty-tree,
      missing-map or single-input limitation in the archive manifest.
- [ ] Install the published artifact once before announcing the release.
- [ ] Protect `main` with the exact required CI checks; reject force pushes and
      branch deletion. Protect `v*` release tags against update/deletion.

Do not tag a feature branch. Prepare a draft release first, merge through a
green pull request, then tag the resulting `main` commit.
