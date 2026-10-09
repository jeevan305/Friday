"""
Unit tests for the git_history tool.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent.repository.workspace import Workspace
from agent.tools.git_history import git_history


def _is_git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


@pytest.fixture
def git_repo(tmp_path: Path):
    """Create a minimal git repository with a commit."""
    if not _is_git_available():
        pytest.skip("git not available")

    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path, capture_output=True,
    )

    f = tmp_path / "Main.java"
    f.write_text("public class Main {}\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path, capture_output=True,
    )
    return tmp_path


def test_git_log_repo(git_repo: Path) -> None:
    ws = Workspace(git_repo)
    result = git_history(ws, operation="log", max_commits=5)

    assert "error" not in result
    assert result["operation"] == "log"
    assert len(result["commits"]) >= 1
    assert "Initial commit" in result["commits"][0]


def test_git_log_file(git_repo: Path) -> None:
    ws = Workspace(git_repo)
    result = git_history(ws, file_path="Main.java", operation="log")

    assert "error" not in result
    assert result["operation"] == "log"
    assert len(result["commits"]) >= 1


def test_git_diff_file(git_repo: Path) -> None:
    ws = Workspace(git_repo)
    result = git_history(ws, file_path="Main.java", operation="diff")

    assert "error" not in result
    assert result["operation"] == "diff"
    assert "Main.java" in result["diff"]


def test_git_invalid_operation(git_repo: Path) -> None:
    ws = Workspace(git_repo)
    result = git_history(ws, operation="foobar")

    assert "error" in result


def test_git_blame_missing_file_arg(git_repo: Path) -> None:
    ws = Workspace(git_repo)
    result = git_history(ws, operation="blame")

    assert "error" in result


def test_git_traversal_blocked(git_repo: Path) -> None:
    ws = Workspace(git_repo)
    result = git_history(ws, file_path="../../etc/passwd", operation="log")

    assert "error" in result
