from __future__ import annotations

import json
from pathlib import Path

from bakenn import __version__
from scripts.build_sbom import build_sbom


def test_cyclonedx_sbom_is_deterministic_and_records_vendored_patches(tmp_path: Path) -> None:
    first = build_sbom(tmp_path / "first.cdx.json")
    second = build_sbom(tmp_path / "second.cdx.json")
    assert first.read_bytes() == second.read_bytes()

    document = json.loads(first.read_text(encoding="utf-8"))
    assert document["bomFormat"] == "CycloneDX"
    assert document["specVersion"] == "1.5"
    assert document["metadata"]["component"]["version"] == __version__
    components = {item["name"]: item for item in document["components"]}
    assert components["CMSIS-NN"]["hashes"][0]["content"]
    cmsis_properties = {
        item["name"]: item["value"] for item in components["CMSIS-NN"]["properties"]
    }
    assert cmsis_properties["bakenn:modified"] == "true"
    assert components["ESP-NN"]["hashes"][0]["content"]
