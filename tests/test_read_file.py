"""
Unit tests for the read_file tool.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.repository.workspace import Workspace
from agent.tools.read_file import read_file


@pytest.fixture
def file_repo(tmp_path: Path) -> Path:
    """Create a small repo with a known file."""
    (tmp_path / "src").mkdir()
    lines = [f"line {i}" for i in range(1, 51)]  # 50 lines
    (tmp_path / "src" / "Sample.java").write_text("\n".join(lines) + "\n")
    (tmp_path / "binary.bin").write_bytes(b"\x00\x01\x02binary")
    return tmp_path


def test_read_full_file(file_repo: Path) -> None:
    ws = Workspace(file_repo)
    result = read_file(ws, "src/Sample.java")

    assert "error" not in result
    assert result["total_lines"] == 50
    assert result["start_line"] == 1
    assert "1 | line 1" in result["content"]


def test_read_line_range(file_repo: Path) -> None:
    ws = Workspace(file_repo)
    result = read_file(ws, "src/Sample.java", start_line=10, end_line=20)

    assert "error" not in result
    assert result["start_line"] == 10
    assert result["end_line"] == 20
    assert "10 | line 10" in result["content"]
    assert "20 | line 20" in result["content"]
    # Lines outside the range should not appear
    assert "line 9" not in result["content"]
    assert "line 21" not in result["content"]


def test_read_file_shows_line_numbers(file_repo: Path) -> None:
    ws = Workspace(file_repo)
    result = read_file(ws, "src/Sample.java", start_line=5, end_line=7)

    lines = result["content"].splitlines()
    assert any("5 |" in l for l in lines)
    assert any("6 |" in l for l in lines)
    assert any("7 |" in l for l in lines)


def test_read_nonexistent_file(file_repo: Path) -> None:
    ws = Workspace(file_repo)
    result = read_file(ws, "nonexistent.java")

    assert "error" in result


def test_read_binary_file(file_repo: Path) -> None:
    ws = Workspace(file_repo)
    result = read_file(ws, "binary.bin")

    assert "error" in result
    assert "binary" in result["error"].lower()


def test_read_start_line_beyond_eof(file_repo: Path) -> None:
    ws = Workspace(file_repo)
    result = read_file(ws, "src/Sample.java", start_line=999)

    assert "error" in result


def test_read_path_traversal(file_repo: Path) -> None:
    ws = Workspace(file_repo)
    result = read_file(ws, "../../etc/passwd")

    assert "error" in result
