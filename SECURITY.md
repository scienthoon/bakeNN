# Security policy

## Supported versions

BakeNN is pre-1.0 alpha software. Security fixes are applied to the latest
`0.1.x` release and the `main` branch; older snapshots are not maintained.

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
