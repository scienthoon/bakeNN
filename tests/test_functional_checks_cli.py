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


def test_test_numbers_are_consecutive_and_titles_unique() -> None:
    completed = _run("--list")
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    identifiers = [line.split()[0] for line in lines]
    assert identifiers
    assert identifiers == [f"T{number:03d}" for number in range(1, len(identifiers) + 1)]
    titles = [line.split("] ", 1)[1] for line in lines]
    assert len(set(titles)) == len(titles)


def test_selected_test_prints_one_result_line_and_a_summary(tmp_path: Path) -> None:
    completed = _run("--strict", "--output", tmp_path / "scratch", "T001")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    lines = completed.stdout.splitlines()
    assert lines[0] == "== install =="
    assert lines[1].startswith("T001 PASS  ")
    assert lines[-1] == "RESULT pass=1 fail=0 skip=0 of 1"


def test_range_selects_every_test_in_it(tmp_path: Path) -> None:
    completed = _run("--strict", "--output", tmp_path / "scratch", "T001-T003", "T005")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    passed = [line.split()[0] for line in completed.stdout.splitlines() if " PASS  " in line]
    assert passed == ["T001", "T002", "T003", "T005"]


@pytest.mark.parametrize("argument", ("T000", "T001-T000", "T003-T001"))
def test_unknown_test_number_is_rejected(argument: str, tmp_path: Path) -> None:
    completed = _run("--output", tmp_path / "scratch", argument)
    assert completed.returncode == 2
    assert f"test id: {argument}" in completed.stderr or f"test range: {argument}" in completed.stderr
    assert not (tmp_path / "scratch").exists()


def test_foreign_output_directory_is_left_alone(tmp_path: Path) -> None:
    keep = tmp_path / "keep.txt"
    keep.write_text("not a scratch directory\n")
    completed = _run("--output", tmp_path, "T001")
    assert completed.returncode == 2
    assert keep.read_text() == "not a scratch directory\n"
