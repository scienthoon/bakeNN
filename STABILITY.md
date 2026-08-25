# Stability policy

BakeNN 1.0.0 is the first stable release of the static batch-one INT8 AOT
compiler contract. This document defines that released compatibility contract
without presenting unfinished roadmap items as released features.

## Versioned deployment contracts

BakeNN follows semantic versioning for its public Python API and the following
independently versioned deployment contracts:

- generated C ABI version `1` (`BKNN_C_ABI_VERSION`);
- generated manifest schema version `4`
  (`BKNN_MANIFEST_SCHEMA_VERSION`);
- arithmetic profile `bakenn.int8.v1`, plus the separately versioned
  `bakenn.softmax_lut.q15.v1` and
  `bakenn.int8.resize_bilinear.q15.v1` numerical contracts, and the
  TFLite-compatible AveragePool profile
  `tflite.int8.average_pool2d.raw_code.v1`;
- versioned kernel implementation and packed-layout IDs recorded in manifests.

Within the BakeNN 1.x line, a compatible update does not silently change the
meaning of C ABI v1, manifest schema v4, or an existing numerical-profile ID.
An incompatible C calling convention or public Python API requires a major
release. An incompatible manifest or arithmetic change also receives a new
schema/profile identifier; readers reject identifiers they do not implement.
Adding a new profile or kernel ID is not permission to reinterpret an old one.

Patch releases preserve these public contracts while fixing defects. Minor
releases may add backward-compatible Python API, kernel IDs, packing IDs, or an
opt-in numerical profile, but C ABI v1 and manifest schema v4 remain the v1
defaults. Removing either contract, making an incompatible schema/profile
mandatory, or dropping an existing public profile requires a major release.

The Python integer reference, portable C, and every enabled optimized backend
must remain byte-exact for the same accepted numerical profile. Numerical
equivalence does not imply that generated source bytes, memory offsets, packed
constants, or kernel choices remain identical across compiler releases.

## Model recompilation policy

Generated libraries are standalone artifacts rather than a target runtime
loaded by BakeNN. A previously validated artifact can remain deployed as-is;
installing a newer host compiler does not mutate it. Recompile and revalidate
the model whenever the model, input shape, qparams/calibration, target,
toolchain or flags, kernel policy, or compiler version used for a new firmware
release changes. Recompilation is also required to pick up a compiler
correctness or security fix.

Do not combine generated sources, headers, manifests, or packed constants from
different compilations. Consumers should check the emitted ABI, manifest
schema, arithmetic-profile ID/version, and artifact hashes before accepting a
generated set. Schema-v4 manifests also carry a canonical payload digest, so a
consumer can reject metadata that was altered independently of the inventoried
generated files.

## Public Python API

The top-level functions and types documented in the README are the intended v1
public API. Internal modules under `bakenn.ir`, `bakenn.plan`, backend
families, and frontend capture types remain compiler internals unless
explicitly documented. Starting with 1.0.0, incompatible public API changes
follow the major-version rule above.

## Intentional v1 product constraints

BakeNN targets firmware in which the MCU and model are fixed, the model is
linked into the application, and model replacement ships with a firmware
rebuild. The following are deliberate v1 constraints:

- batch size one and fully static shapes;
- one public model input and one public model output;
- no runtime model loader, dynamic tensor allocation, or target-side float
  fallback;
- a narrower, fail-closed operator surface than TFLite Micro.

Internal graphs may still contain residual branches, concatenation, SE
broadcast operations, and other multi-input nodes. The single-input/output
constraint applies to the public firmware ABI.

## Support window

Security and correctness fixes are applied to the latest supported 1.x release
and `main`. Development snapshots are not maintained as separate release
lines. Report security issues via [SECURITY.md](SECURITY.md) and compatibility
requests using the repository's issue templates.
