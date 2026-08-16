from __future__ import annotations

import os
from pathlib import Path

from scripts.verify_mnist_evidence import verify


def test_frozen_mnist_evidence_compiles_and_matches_expected_bytes() -> None:
    root = Path(__file__).resolve().parents[1]
    result = verify(root / "examples/mnist/evidence", os.environ.get("CC", "cc"))

    assert result["result"] == "PASS"
    assert result["physical_measurement"] is False
    assert result["verified_payload_files"] >= 17
    assert result["compared_output_bytes"] == 1000
    assert result["mismatched_output_bytes"] == 0
    assert result["output_fnv1a"] == "0x55fb9e60"
    assert result["output_sha256"] == (
        "cdbbfc38bf5f17a2867b17a8a28e643043e50733063e246b0bc85cbeaae30ba4"
    )

