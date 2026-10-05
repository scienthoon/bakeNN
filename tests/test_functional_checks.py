from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


def test_functional_checks_pass_offline(tmp_path: Path) -> None:
    """The scripted functional checks must keep passing from a plain checkout."""

    pytest.importorskip("torch")
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(repository / "scripts/functional_checks.py"), "--output", str(tmp_path / "run")],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert " FAIL " not in completed.stdout
    assert "RESULT pass=" in completed.stdout
