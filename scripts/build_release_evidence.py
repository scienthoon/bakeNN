#!/usr/bin/env python3
"""Build a deterministic archive of BakeNN's physical benchmark evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import zipfile


REPOSITORY = Path(__file__).resolve().parents[1]
ARCHIVE_TIME = (1980, 1, 1, 0, 0, 0)
FIXED_FILES = (
    "CHANGELOG.md",
    "LICENSE",
    "README.md",
    "REPRODUCING.md",
    "STABILITY.md",
    "THIRD_PARTY_NOTICES.md",
    ".github/release-notes/v0.1.0.md",
    "benchmarks/RESULTS.md",
    "benchmarks/esp32/README.md",
    "benchmarks/tflm_compare/README.md",
)
EVIDENCE_DIRECTORIES = (
    "benchmarks/esp32/results",
    "benchmarks/tflm_compare/results",
)


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _version() -> str:
    source = (REPOSITORY / "src/bakenn/_version.py").read_text(encoding="utf-8")
    match = re.search(r'^VERSION = "([^"]+)"$', source, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError("could not read BakeNN version")
    return match.group(1)


def _archive_name(label: str) -> str:
    path = PurePosixPath(label)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe artifact label: {label}")
    if any(part in ("", ".") for part in path.parts):
        raise ValueError(f"unsafe artifact label: {label}")
    return str(PurePosixPath("artifacts", path))


def _tracked_evidence() -> dict[str, bytes]:
    entries: dict[str, bytes] = {}
    for relative in FIXED_FILES:
        path = REPOSITORY / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        entries[relative] = path.read_bytes()
    for relative in EVIDENCE_DIRECTORIES:
        directory = REPOSITORY / relative
        for path in sorted(directory.iterdir()):
            if path.is_file() and not path.name.startswith("."):
                name = path.relative_to(REPOSITORY).as_posix()
                entries[name] = path.read_bytes()
    return entries


def build_archive(output: Path, artifact_specs: tuple[str, ...]) -> tuple[Path, str]:
    entries = _tracked_evidence()
    external: list[dict[str, object]] = []
    for specification in artifact_specs:
        if "=" not in specification:
            raise ValueError("--artifact must use LABEL=PATH")
        label, raw_path = specification.split("=", 1)
        source = Path(raw_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        archive_name = _archive_name(label)
        if archive_name in entries:
            raise ValueError(f"duplicate archive path: {archive_name}")
        data = source.read_bytes()
        entries[archive_name] = data
        external.append(
            {
                "archive_path": archive_name,
                "source_name": source.name,
                "bytes": len(data),
                "sha256": _sha256(data),
            }
        )

    commit = _git("rev-parse", "HEAD")
    tag = _git("tag", "--points-at", "HEAD").splitlines()
    dirty = bool(_git("status", "--porcelain"))
    manifest = {
        "schema_version": 1,
        "bakenn_version": _version(),
        "source_commit": commit,
        "source_tags": sorted(value for value in tag if value),
        "working_tree_dirty": dirty,
        "scope": (
            "Checked-in physical benchmark reports and explicitly supplied "
            "binary/linker artifacts; missing metrics remain unmeasured."
        ),
        "external_artifacts": external,
        "files": [
            {"path": name, "bytes": len(data), "sha256": _sha256(data)}
            for name, data in sorted(entries.items())
        ],
    }
    entries["MANIFEST.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(name, ARCHIVE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    digest = _sha256(output.read_bytes())
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return output, digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="include an exact ELF, map, UART or other physical artifact",
    )
    arguments = parser.parse_args()
    output, digest = build_archive(arguments.output, tuple(arguments.artifact))
    print(f"{output} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
