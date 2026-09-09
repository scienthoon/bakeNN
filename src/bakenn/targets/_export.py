"""Publish complete target projects without overwriting application files."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterator

from bakenn.errors import CompileError


def _destination_state(output: Path) -> tuple[int, int] | None:
    if output.is_symlink():
        raise CompileError(f"export output must not be a symlink: {output}")
    if not output.exists():
        return None
    if not output.is_dir():
        raise CompileError(f"export output is not a directory: {output}")
    if any(output.iterdir()):
        raise CompileError(f"refusing to overwrite non-empty export directory: {output}")
    stat = output.stat()
    return stat.st_dev, stat.st_ino


@contextmanager
def staged_export(source: Path, output: Path) -> Iterator[Path]:
    """Require a disjoint, absent/empty output and publish only after success."""

    staging: Path | None = None
    try:
        source_root = source.resolve()
        destination = output.resolve()
        if destination == source_root or source_root in destination.parents:
            raise CompileError("export destination must not be inside the source artifacts")
        state = _destination_state(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(
            prefix=f".{output.name}.bakenn-export-", dir=output.parent
        ))
        yield staging
        if _destination_state(output) != state:
            raise CompileError(f"export output changed concurrently: {output}")
        # A directory rename can replace an empty directory atomically; do not
        # delete it first and create a window with a missing user destination.
        os.replace(staging, output)
    except OSError as error:
        raise CompileError(f"failed to export target project: {error}") from error
    finally:
        if staging is not None and staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging, ignore_errors=True)
