from __future__ import annotations

import os
import shutil

import pytest


def require_compiler(compiler: str) -> str:
    """Resolve a compiler and honor the CI fail-closed compiler contract."""

    resolved = shutil.which(compiler)
    if resolved is not None:
        return resolved
    message = f"required C compiler is unavailable: {compiler}"
    if os.environ.get("BAKENN_REQUIRE_CC") == "1":
        pytest.fail(message)
    pytest.skip(message)


__all__ = ["require_compiler"]
