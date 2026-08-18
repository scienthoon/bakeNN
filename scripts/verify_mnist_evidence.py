#!/usr/bin/env python3
"""Dependency-free clean-room verification of the frozen MNIST C artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = REPOSITORY / "examples/mnist/evidence"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fnv1a(data: bytes) -> str:
    value = 2166136261
    for item in data:
        value ^= item
        value = (value * 16777619) & 0xFFFFFFFF
    return f"0x{value:08x}"


def verify(evidence_dir: Path, compiler: str) -> dict[str, object]:
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        raise RuntimeError(f"C compiler not found: {compiler}")
    manifest_path = evidence_dir / "mnist_evidence.json"
    evidence = json.loads(manifest_path.read_text(encoding="utf-8"))

    verified_files: list[dict[str, object]] = []
    for item in evidence["files"]:
        path = evidence_dir / item["path"]
        actual = _sha256(path)
        if actual != item["sha256"]:
            raise RuntimeError(
                f"hash mismatch for {item['path']}: {actual} != {item['sha256']}"
            )
        if path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"byte-count mismatch for {item['path']}")
        verified_files.append(
            {"path": item["path"], "bytes": item["bytes"], "sha256": actual}
        )

    generated = evidence_dir / "generated"
    deployment = json.loads(
        (generated / "bknn_mnist_manifest.json").read_text(encoding="utf-8")
    )
    symbol = str(deployment["model"])
    macro = symbol.upper()
    with tempfile.TemporaryDirectory(prefix="bakenn-mnist-clean-room-") as temporary:
        temporary_dir = Path(temporary)
        runner = temporary_dir / "runner.c"
        runner.write_text(
            f'''#include "{symbol}.h"
#include <stddef.h>
#include <stdio.h>

#define ARENA_STORAGE_SIZE ({macro}_ARENA_SIZE == 0u ? 1u : {macro}_ARENA_SIZE)

int main(void) {{
    _Alignas({macro}_ARENA_ALIGNMENT) static unsigned char arena[ARENA_STORAGE_SIZE];
    static signed char input[{macro}_INPUT_SIZE];
    static signed char output[{macro}_OUTPUT_SIZE];
    unsigned char *arena_ptr = {macro}_ARENA_SIZE == 0u ? NULL : arena;
    while (fread(input, 1u, {macro}_INPUT_BYTES, stdin) == {macro}_INPUT_BYTES) {{
        {symbol}_infer(arena_ptr, input, output);
        if (fwrite(output, 1u, {macro}_OUTPUT_BYTES, stdout) != {macro}_OUTPUT_BYTES) {{
            return 2;
        }}
    }}
    return ferror(stdin) ? 3 : 0;
}}
''',
            encoding="utf-8",
        )
        executable = temporary_dir / "mnist_clean_room"
        command = (
            compiler_path,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            str(generated / "bknn_mnist.c"),
            str(generated / "bknn_mnist_weights.c"),
            str(generated / "bknn_mnist_kernels.c"),
            str(runner),
            "-I",
            str(generated),
            "-o",
            str(executable),
        )
        subprocess.run(command, check=True, capture_output=True)
        input_bytes = (evidence_dir / "physical_test_inputs_int8.bin").read_bytes()
        expected = (evidence_dir / "physical_expected_outputs_int8.bin").read_bytes()
        completed = subprocess.run(
            (str(executable),), input=input_bytes, check=True, capture_output=True
        )

    actual = completed.stdout
    compared = min(len(actual), len(expected))
    mismatches = sum(left != right for left, right in zip(actual, expected))
    mismatches += abs(len(actual) - len(expected))
    if actual != expected:
        raise RuntimeError(
            f"generated C output mismatch: bytes={len(actual)}/{len(expected)}, "
            f"mismatches={mismatches}"
        )
    compiler_version = subprocess.run(
        (compiler_path, "--version"), check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    return {
        "schema_version": 1,
        "claim": "independent host clean-room artifact reproduction",
        "physical_measurement": False,
        "source_commit_recorded_by_evidence": evidence["source_commit"],
        "mnist_evidence_manifest_sha256": _sha256(manifest_path),
        "checkpoint_file_sha256": evidence["checkpoint"]["file_sha256"],
        "checkpoint_logical_sha256": evidence["checkpoint"][
            "logical_tensor_sha256"
        ],
        "calibration_corpus_sha256": evidence["calibration"]["corpus_sha256"],
        "generated_artifact_set_sha256": evidence["generated_artifacts"][
            "set_sha256"
        ],
        "verified_payload_files": len(verified_files),
        "compiler": compiler_version,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "compared_output_bytes": compared,
        "mismatched_output_bytes": mismatches,
        "output_sha256": hashlib.sha256(actual).hexdigest(),
        "output_fnv1a": _fnv1a(actual),
        "environment": {
            "cc_requested": compiler,
            "ci": os.environ.get("CI") == "true",
        },
        "result": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument(
        "--output",
        type=Path,
        help="write the machine-readable reproduction result for a PR",
    )
    arguments = parser.parse_args()
    result = verify(arguments.evidence_dir, arguments.cc)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
