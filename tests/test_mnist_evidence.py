from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.verify_mnist_evidence import verify


def test_frozen_mnist_evidence_compiles_and_matches_expected_bytes() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence_dir = root / "examples/mnist/evidence"
    evidence = json.loads(
        (evidence_dir / "mnist_evidence.json").read_text(encoding="utf-8")
    )
    result = verify(evidence_dir, os.environ.get("CC", "cc"))

    assert evidence["training"]["performed"] is True
    assert evidence["training"]["epochs"] == 4
    assert evidence["working_tree_dirty"] is False
    assert len(evidence["source_commit"]) == 40
    assert result["result"] == "PASS"
    assert result["physical_measurement"] is False
    assert result["verified_payload_files"] >= 17
    assert result["compared_output_bytes"] == 1000
    assert result["mismatched_output_bytes"] == 0
    assert result["output_fnv1a"] == "0x55fb9e60"
    assert result["output_sha256"] == (
        "cdbbfc38bf5f17a2867b17a8a28e643043e50733063e246b0bc85cbeaae30ba4"
    )
