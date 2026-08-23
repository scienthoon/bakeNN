#!/usr/bin/env python3
"""Create a deterministic CycloneDX SBOM for one BakeNN source revision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


REPOSITORY = Path(__file__).resolve().parents[1]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tree_hash(relative: str) -> str:
    root = REPOSITORY / relative
    digest = hashlib.sha256()
    for path in sorted(
        item
        for item in root.rglob("*")
        if item.is_file()
        and "__pycache__" not in item.parts
        and item.suffix not in {".pyc", ".pyo"}
    ):
        name = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        payload_hash = bytes.fromhex(_sha256(path.read_bytes()))
        digest.update(payload_hash)
    return digest.hexdigest()


def _version() -> str:
    source = (REPOSITORY / "src/bakenn/_version.py").read_text(encoding="utf-8")
    match = re.search(r'^VERSION = "([^"]+)"$', source, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError("could not read BakeNN version")
    return match.group(1)


def _component(
    name: str,
    version: str,
    *,
    scope: str,
    license_id: str,
    purl: str | None = None,
    source_hash: str | None = None,
    properties: tuple[tuple[str, str], ...] = (),
) -> dict[str, object]:
    reference = f"pkg:{name}@{version}"
    result: dict[str, object] = {
        "type": "library",
        "bom-ref": reference,
        "name": name,
        "version": version,
        "scope": scope,
        "licenses": [{"license": {"id": license_id}}],
        "properties": [
            {"name": key, "value": value} for key, value in properties
        ],
    }
    if purl is not None:
        result["purl"] = purl
    if source_hash is not None:
        result["hashes"] = [{"alg": "SHA-256", "content": source_hash}]
    return result


def build_sbom(output: Path) -> Path:
    version = _version()
    root_ref = f"pkg:pypi/bakenn@{version}"
    components = [
        _component(
            "numpy",
            ">=1.24,<3",
            scope="required",
            license_id="BSD-3-Clause",
            purl="pkg:pypi/numpy",
        ),
        _component(
            "torch",
            ">=2.3,<2.11",
            scope="optional",
            license_id="BSD-3-Clause",
            purl="pkg:pypi/torch",
            properties=(("bakenn:extra", "torch,model-zoo"),),
        ),
        _component(
            "torchvision",
            ">=0.18,<0.26",
            scope="optional",
            license_id="BSD-3-Clause",
            purl="pkg:pypi/torchvision",
            properties=(("bakenn:extra", "model-zoo"),),
        ),
        _component(
            "flatbuffers",
            ">=23,<26",
            scope="optional",
            license_id="Apache-2.0",
            purl="pkg:pypi/flatbuffers",
            properties=(("bakenn:extra", "tflite"), ("bakenn:host_only", "true")),
        ),
        _component(
            "tflite",
            ">=2.18,<3",
            scope="optional",
            license_id="Apache-2.0",
            purl="pkg:pypi/tflite",
            properties=(("bakenn:extra", "tflite"), ("bakenn:host_only", "true")),
        ),
        _component(
            "ai-edge-litert",
            ">=2.2,<3",
            scope="optional",
            license_id="Apache-2.0",
            purl="pkg:pypi/ai-edge-litert",
            properties=(
                ("bakenn:extra", "tflite-verify"),
                ("bakenn:test_oracle", "true"),
                ("bakenn:host_only", "true"),
            ),
        ),
        _component(
            "CMSIS-NN",
            "4.0.0+ca5dc343",
            scope="optional",
            license_id="Apache-2.0",
            source_hash=_tree_hash("src/bakenn/backend/cmsis_nn/vendor/cmsis_nn"),
            properties=(
                ("bakenn:vendored", "true"),
                ("bakenn:modified", "true"),
                ("bakenn:provenance", "src/bakenn/backend/cmsis_nn/vendor/cmsis_nn/BAKENN_PROVENANCE.md"),
            ),
        ),
        _component(
            "CMSIS-Core",
            "5.9.0",
            scope="optional",
            license_id="Apache-2.0",
            source_hash=_tree_hash("src/bakenn/backend/cmsis_nn/vendor/cmsis_core"),
            properties=(("bakenn:vendored", "true"),),
        ),
        _component(
            "ESP-NN",
            "1.2.6+c0876179",
            scope="optional",
            license_id="Apache-2.0",
            source_hash=_tree_hash("src/bakenn/backend/esp_nn/vendor/esp_nn"),
            properties=(
                ("bakenn:vendored", "true"),
                ("bakenn:modified", "false"),
                ("bakenn:provenance", "src/bakenn/backend/esp_nn/vendor/esp_nn/BAKENN_PROVENANCE.md"),
            ),
        ),
    ]
    document = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{_sha256(root_ref.encode())[:8]}-{_sha256(root_ref.encode())[8:12]}-5{_sha256(root_ref.encode())[13:16]}-a{_sha256(root_ref.encode())[17:20]}-{_sha256(root_ref.encode())[20:32]}",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "bom-ref": root_ref,
                "name": "bakenn",
                "version": version,
                "purl": root_ref,
                "licenses": [{"license": {"id": "Apache-2.0"}}],
                "hashes": [
                    {
                        "alg": "SHA-256",
                        "content": _tree_hash("src/bakenn"),
                    }
                ],
            }
        },
        "components": components,
        "dependencies": [
            {
                "ref": root_ref,
                "dependsOn": [item["bom-ref"] for item in components],
            },
            *({"ref": item["bom-ref"], "dependsOn": []} for item in components),
        ],
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(build_sbom(arguments.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
