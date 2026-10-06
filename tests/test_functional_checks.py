"""The numbered functional tests T01, T02, ... as one pytest case each."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/functional_checks.py"
_spec = importlib.util.spec_from_file_location("functional_checks", SCRIPT)
assert _spec is not None and _spec.loader is not None
functional_checks = importlib.util.module_from_spec(_spec)
# Registered before execution so the script's dataclasses resolve their module.
sys.modules["functional_checks"] = functional_checks
_spec.loader.exec_module(functional_checks)


@pytest.fixture(scope="module")
def session(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    output = tmp_path_factory.mktemp("functional_checks")
    return functional_checks.Session(output, os.environ.get("CC", "cc"))


@pytest.mark.parametrize(
    "check", functional_checks.CHECKS, ids=[check.identifier for check in functional_checks.CHECKS]
)
def test_functional_check(check, session) -> None:  # type: ignore[no-untyped-def]
    """Each numbered test passes, or is skipped only for a missing host tool."""

    try:
        detail = functional_checks.run_check(check, session)
    except functional_checks.Skip as skipped:
        pytest.skip(str(skipped))
    assert detail
