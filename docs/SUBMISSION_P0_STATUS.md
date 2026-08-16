# Submission P0 status

This file is the single checklist for the evidence requested before the
competition submission. A checked item means the evidence is committed and
reproducible; it does not turn a boardless result into a physical measurement.

## Completed in the repository

- [x] Train the MNIST CNN from FP32 for four epochs and compile the full model
  with PTQ to standalone INT8 C.
- [x] Record FP32 accuracy, generated-C INT8 accuracy, and Python integer
  reference versus generated-C byte agreement.
- [x] Store the trained checkpoint, calibration corpus, physical input corpus,
  expected outputs, generated sources, manifests, and SHA-256 hashes under
  `examples/mnist/evidence/`.
- [x] Provide the one-command, standard-library clean-room verifier at
  `scripts/verify_mnist_evidence.py`.
- [x] Document an evidence-backed comparison with TFLite Micro and Edge
  Impulse EON in `docs/COMPARISON.md`.
- [x] Keep UART/cycle/Flash/SRAM measurements from physical MCUs under
  `benchmarks/physical/` and keep compiler/linker-only results under
  `benchmarks/cross_build/`.
- [x] Compare the same trained-MNIST quantized contract against Apache TVM
  0.16 AOT + USMP + CMSIS-NN for Cortex-M4, including generated sources,
  cross-linked ELF resource sizes, and 1,000-byte output agreement.
- [x] Integrate the results and evidence boundaries into the competition DOCX
  and PDF report.

## External gates that cannot be self-certified

- [ ] **Independent clean-room PR.** A contributor other than the project
  author must clone the public repository, run the verifier, and submit the
  resulting JSON in a pull request. Instructions and a PR-ready template are in
  `docs/CLEAN_ROOM_REPRODUCTION.md` and `reproductions/README.md`.
- [ ] **Trained-MNIST physical benchmark.** The ESP32 project, fixed 100-image
  corpus, expected output, hashes, and acceptance line are ready, but the UART
  transcript, device cycles, and stack watermark must come from an actual
  board run before this result may appear in the physical benchmark table.

These two boxes are deliberately not replaceable by a local simulation,
cross-build, or a pull request authored by the maintainer.
