# Three-minute demo script

1. Show `DemoCNN` in `generate.py`: an ordinary FP32 PyTorch model.
2. Run the generation command and point out the selected ESP-NN Conv,
   Depthwise and FullyConnected kernel IDs in `demo_summary.json`.
3. Open `generated/bknn_esp32s3_demo_cnn.c` to show the fixed execution order
   and static arena offsets; open the manifest for qparams and resource sizes.
4. Run `idf.py build` to prove that the self-contained project links for the
   ESP32-S3 toolchain.
5. If a board is available, flash it and show cold/median/p95 cycles, stack
   watermark and output checksum from the generated UART runner.

Be explicit that the demo model is untrained and that boardless compilation is
not a latency result. Use the checked-in nRF52840 report for measured BakeNN
versus TFLM claims.
