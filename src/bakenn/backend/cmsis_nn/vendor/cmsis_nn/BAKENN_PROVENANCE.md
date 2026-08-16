# Vendored CMSIS-NN provenance

- Upstream: https://github.com/ARM-software/CMSIS-NN
- Release version: 4.0.0
- Git revision: `ca5dc34313be2ee5c46652917c30baac96c52621`
- License: Apache-2.0; see `LICENSE.txt` in this directory

BakeNN vendors only the source and headers needed by the selected direct
CMSIS-NN operators.  The upstream files are not modified.  BakeNN's lowering,
capability predicates, static scratch planning and bundle selection live
outside this directory.
