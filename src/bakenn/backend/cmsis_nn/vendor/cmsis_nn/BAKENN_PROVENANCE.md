# Vendored CMSIS-NN provenance

- Upstream: <https://github.com/ARM-software/CMSIS-NN>
- Release version: 4.0.0
- Git revision: `ca5dc34313be2ee5c46652917c30baac96c52621`
- License: Apache-2.0; see `LICENSE.txt` in this directory

BakeNN vendors only the source and headers required by the selected direct
CMSIS-NN operators. Most copied files are byte-identical to that revision. The
following files carry prominent BakeNN modification notices and deliberately
differ from upstream:

| File | Upstream SHA-256 | Vendored SHA-256 | Purpose |
|---|---|---|---|
| `Include/arm_nn_math_types.h` | `c5bbdf59d6bb98ae0ac58bcffda25d812bac71790b53508a9268035aae84d8ec` | `3f3c21e68b0fb07af8f43765477cc69a04f538a1eecd152d6d9caab331edae91` | Gated freestanding memory shim and removal of hosted math headers in that mode |
| `Source/ConvolutionFunctions/arm_convolve_1x1_s8.c` | `4039c6246f27f3a805817b68073e4f9a599b67cebc647b5fc8b1ca8be9dbdb4c` | `f805ed70d7a43a24069d6fb86ae48c68b61e3596fa55dd65c107e84b9948025c` | Remove unused hosted `stdio.h` include |
| `Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.c` | `0621c7780076974673eb55d07c6ef195b8bae42640b01c9c308a2249f0ad66f9` | `c45e80f5d472adc6bcfdef6269d38120db17d1ea7b23e99eab26e2c334b0fe99` | Remove unused hosted `stdio.h` include |
| `Source/ConvolutionFunctions/arm_depthwise_conv_s8.c` | `5777348073148c1d3f6a353bda21847320806c0f08044b3e4810230a39498944` | `403a9bb753a55c9be1f43b0fb7b558999c7c1c55152a0d90ea30571c004e55e2` | Limit a GCC-only optimize attribute to compatible compilers |

BakeNN also adds
`Source/NNSupportFunctions/bakenn_cmsis_memory.c` (SHA-256
`4c08e36d24eb66bc1ad4b3351a81dda34b9105d60b523d64acf96690a3063e7e`)
as an Apache-2.0 freestanding support file; it is not represented as an
upstream CMSIS-NN file.

The compiler-side lowering, capability predicates, static scratch planning and
bundle selection live outside this vendor directory. Generated bundles retain
this record and the packaged upstream license so downstream firmware can audit
and redistribute the selected source closure accurately.
