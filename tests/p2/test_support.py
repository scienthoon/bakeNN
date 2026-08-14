from __future__ import annotations

import pytest

from . import support


def test_required_compiler_is_a_failure_instead_of_a_skip(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("BAKENN_REQUIRE_CC", "1")
    monkeypatch.setattr(support.shutil, "which", lambda _: None)
    with pytest.raises(pytest.fail.Exception, match="required C compiler"):
        support.require_compiler("missing-compiler")
