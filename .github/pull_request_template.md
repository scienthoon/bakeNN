## What changed

Describe the scoped change and why it belongs in BakeNN's fixed-model contract.

## Numerical and fallback contract

State qparams/rounding/layout assumptions, capability predicates, fallback or
fail-closed behavior, and any generated-code/resource impact.

## Verification

- [ ] Python reference and generated C are byte-exact.
- [ ] Malformed and unsupported cases are tested.
- [ ] GCC/Clang strict C and applicable ASan/UBSan checks pass.
- [ ] Applicable cross-toolchain/target builds pass.
- [ ] Performance claims include same-target raw evidence, or no performance
      claim is made.
- [ ] Documentation and vendored license/provenance files are updated.
