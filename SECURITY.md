# Security policy

## Supported versions

BakeNN follows the 1.x compatibility and support policy in `STABILITY.md`.
Security fixes are applied to the latest supported `1.x` release and the
`main` branch; older development snapshots are not maintained.

## Trust boundaries

Generated artifact manifests use SHA-256 digests to detect file corruption and
inconsistent artifact sets. These are unkeyed integrity checks, not digital
signatures: someone who can replace both files and their manifest can recompute
the hashes. Authenticate the artifact source separately before building or
deploying externally supplied C code.

PyTorch capture runs the supplied Python model on the host. Only capture model
code you trust; it is not a sandbox for untrusted Python. Static IR checks and
the optional TFLite importer validate their supported model representation,
but do not provide host process isolation or resource limits.

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could cause generated-code
memory corruption, unsafe compilation, artifact tampering or dependency
substitution. Use GitHub's **Security -> Report a vulnerability** flow for this
repository. Include:

- the affected BakeNN revision and host/target toolchain;
- a minimal model, graph or generated artifact;
- expected and observed behavior;
- whether ASan, UBSan, a cross-compiler or physical hardware reproduced it;
- any known impact and suggested embargo constraints.

The maintainer will acknowledge a complete report as soon as practical,
validate its scope, prepare a fix and coordinate disclosure. No specific
response deadline is guaranteed for this volunteer project.

Numerical mismatches without a security impact may be reported through the bug
template, but always include raw INT8 inputs and expected/actual output bytes.
