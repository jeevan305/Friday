"""
Unit tests for the list_directory tool.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from agent.repository.workspace import Workspace, PathTraversalError
from agent.tools.list_directory import list_directory


@pytest.fixture
def temp_repo(tmp_path: Path) -> Path:
    """Create a small fake repository layout."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "Main.java").write_text("// main")
    (tmp_path / "src" / "Helper.java").write_text("// helper")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "README.md").write_text("# readme")
    (tmp_path / "README.md").write_text("# root")
    # Git dir should be skipped
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]")
    return tmp_path


def test_list_root(temp_repo: Path) -> None:
    ws = Workspace(temp_repo)
    result = list_directory(ws, ".")

    assert "error" not in result
    assert "src" in result["directories"]
    assert "docs" in result["directories"]
    assert ".git" not in result["directories"]  # pruned
    assert "README.md" in result["files"]


def test_list_subdirectory(temp_repo: Path) -> None:
    ws = Workspace(temp_repo)
    result = list_directory(ws, "src")

    assert "error" not in result
    assert result["directories"] == []
    assert "Main.java" in result["files"]
    assert "Helper.java" in result["files"]


def test_list_nonexistent_path(temp_repo: Path) -> None:
    ws = Workspace(temp_repo)
    result = list_directory(ws, "does/not/exist")

    assert "error" in result


def test_list_file_not_directory(temp_repo: Path) -> None:
    ws = Workspace(temp_repo)
    result = list_directory(ws, "README.md")

    assert "error" in result


def test_path_traversal_blocked(temp_repo: Path) -> None:
    ws = Workspace(temp_repo)
    result = list_directory(ws, "../../etc")

    assert "error" in result
    assert "escapes" in result["error"].lower() or "access denied" in result["error"].lower()
