"""Deterministic provenance helpers for the trained MNIST example."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _update_field(digest: "hashlib._Hash", value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="little", signed=False))
    digest.update(value)


def logical_checkpoint_sha256(state_dict: Mapping[str, object]) -> str:
    """Hash tensor names, dtypes, shapes and bytes independent of torch.save."""

    digest = hashlib.sha256(b"bakenn.mnist.checkpoint.logical.v1\0")
    for name in sorted(state_dict):
        value = state_dict[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()  # type: ignore[union-attr]
        array = np.ascontiguousarray(np.asarray(value))
        if array.dtype.hasobject:
            raise TypeError(f"checkpoint tensor {name!r} has object dtype")
        _update_field(digest, name.encode("utf-8"))
        _update_field(digest, array.dtype.str.encode("ascii"))
        _update_field(
            digest,
            json.dumps(array.shape, separators=(",", ":")).encode("ascii"),
        )
        _update_field(digest, array.tobytes(order="C"))
    return digest.hexdigest()


def corpus_sha256(images: np.ndarray, labels: np.ndarray, *, domain: str) -> str:
    images = np.ascontiguousarray(images)
    labels = np.ascontiguousarray(labels)
    digest = hashlib.sha256(f"bakenn.mnist.{domain}.v1\0".encode("ascii"))
    for array in (images, labels):
        _update_field(digest, array.dtype.str.encode("ascii"))
        _update_field(
            digest,
            json.dumps(array.shape, separators=(",", ":")).encode("ascii"),
        )
        _update_field(digest, array.tobytes(order="C"))
    return digest.hexdigest()


def artifact_set_sha256(root: Path, paths: list[Path]) -> str:
    """Hash a named file set so generated artifacts cannot be mixed."""

    digest = hashlib.sha256(b"bakenn.mnist.generated-artifact-set.v1\0")
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        _update_field(digest, relative)
        _update_field(digest, path.read_bytes())
    return digest.hexdigest()


def file_record(path: Path, *, root: Path, role: str) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


__all__ = [
    "artifact_set_sha256",
    "corpus_sha256",
    "file_record",
    "logical_checkpoint_sha256",
    "sha256_file",
]
