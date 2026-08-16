# Independent clean-room reproduction

The submission gate requires at least one pull request authored by someone who
did not produce BakeNN's implementation or checked-in benchmark result. A PR
from the maintainer's second branch, machine or account is not independent.

The reproducer does not need PyTorch, torchvision, MNIST downloads or an MCU.
The frozen evidence already contains the checkpoint, calibration corpus,
generated C and expected INT8 bytes. The reproducer independently verifies all
payload hashes, compiles the generated C with strict warnings and executes 100
inputs through it.

## Reproducer steps

```bash
git clone https://github.com/scienthoon/bakeNN.git
cd bakeNN
git checkout main
python scripts/verify_mnist_evidence.py \
  --output reproductions/YOUR_GITHUB_HANDLE.json
```

Run it once with GCC or Clang. The final fields must include:

```text
result: PASS
compared_output_bytes: 1000
mismatched_output_bytes: 0
output_fnv1a: 0x55fb9e60
```

Then commit only `reproductions/YOUR_GITHUB_HANDLE.json` and open a pull
request. In the PR body state:

1. operating system and whether the clone was fresh;
2. whether the author had prior access to uncommitted BakeNN source/results;
3. the exact command used;
4. whether any repository file was modified to make the command pass.

CI reruns the verifier and validates the submitted result. This is host
clean-room evidence, not a physical-board performance measurement.

