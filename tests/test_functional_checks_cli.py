"""The command-line entry point of ``scripts/functional_checks.py``."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/functional_checks.py"


def _run(*arguments: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, arguments)], capture_output=True, text=True
    )


def test_test_numbers_are_consecutive() -> None:
    completed = _run("--list")
    assert completed.returncode == 0, completed.stderr
    identifiers = [line.split()[0] for line in completed.stdout.splitlines()]
    assert identifiers
    assert identifiers == [f"T{number:02d}" for number in range(1, len(identifiers) + 1)]


def test_selected_test_prints_one_result_line_and_a_summary(tmp_path: Path) -> None:
    completed = _run("--strict", "--output", tmp_path / "scratch", "T01")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    lines = completed.stdout.splitlines()
    assert lines[0].startswith("T01 PASS  ")
    assert lines[-1] == "RESULT pass=1 fail=0 skip=0 of 1"


def test_unknown_test_number_is_rejected(tmp_path: Path) -> None:
    completed = _run("--output", tmp_path / "scratch", "T00")
    assert completed.returncode == 2
    assert "unknown test id: T00" in completed.stderr
    assert not (tmp_path / "scratch").exists()


def test_foreign_output_directory_is_left_alone(tmp_path: Path) -> None:
    keep = tmp_path / "keep.txt"
    keep.write_text("not a scratch directory\n")
    completed = _run("--output", tmp_path, "T01")
    assert completed.returncode == 2
    assert keep.read_text() == "not a scratch directory\n"
