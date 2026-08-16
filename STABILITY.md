# Stability policy

BakeNN 0.1.x is alpha software.  The compiler is usable for the documented
static INT8 contract, but the Python API and generated artifact layout have not
reached 1.0 compatibility guarantees.

## Compatibility tiers

### Versioned numerical contracts

The following identifiers describe deployment semantics rather than an
implementation detail.  BakeNN will not silently change their arithmetic:

- `bakenn.int8.v1` for Q31 requantization and integer operator behavior;
- `bakenn.softmax_lut.q15.v1` for the current Softmax LUT contract;
- `bakenn.int8.resize_bilinear.q15.v1` for bilinear resize coordinates and
  rounding;
- versioned kernel implementation and packed-layout IDs recorded in manifests.

A future incompatible numerical rule receives a new identifier.  Within the
same identifier, the Python integer reference, portable C and every enabled
optimized backend must remain byte-exact.

### Public Python API

Top-level functions documented in the README are the intended 0.1 public API,
but pre-1.0 minor releases may rename or reorganize them with changelog and
migration notes.  Internal modules under `bakenn.ir`, `bakenn.plan`, backend
families and frontend capture types are compiler internals unless explicitly
documented otherwise.

### Generated C and manifests

Generated symbols use `bknn_`/`BKNN_`, but exact filenames, helper names,
manifest fields, memory-plan offsets and selected kernels may change between
minor releases.  Recompile the model when upgrading BakeNN.  Do not link C
artifacts produced by different BakeNN versions into one ABI contract unless
their public headers are independently reviewed.

## Intentional product constraints

BakeNN targets firmware in which the MCU and model are fixed, the model is
linked into the application, and model replacement ships with a firmware
rebuild.  The following are intentional 0.1 constraints:

- batch size one and fully static shapes;
- one public model input and one public model output;
- no runtime model loader, dynamic tensor allocation or target-side float
  fallback;
- a narrower fail-closed operator surface than TFLite Micro.

Internal graphs may still contain residual branches, concatenation, SE
broadcast operations and other multi-input nodes.  The single-input/output
constraint applies to the public firmware ABI.

## Support window

Security and correctness fixes are applied to the latest 0.1.x release and
`main`.  Older alpha snapshots are not maintained.  Report security issues via
[SECURITY.md](SECURITY.md) and compatibility requests using the repository's
issue templates.
