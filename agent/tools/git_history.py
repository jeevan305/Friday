"""
git_history tool — inspect Git history for investigation context.

Supports:
  - recent commits for a file (log)
  - git blame for specific lines
  - recent changes diff
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

from agent.repository.workspace import Workspace, PathTraversalError


def _git_available() -> bool:
    return shutil.which("git") is not None


def _run_git(args: list[str], cwd: str, timeout: int = 15) -> tuple[str, str, int]:
    """Run a git command; returns (stdout, stderr, returncode)."""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "git command timed out", 1
    except FileNotFoundError:
        return "", "git not found", 1


def git_history(
    workspace: Workspace,
    file_path: str | None = None,
    operation: str = "log",
    max_commits: int = 10,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict[str, Any]:
    """
    Inspect Git history for investigation context.

    Parameters
    ----------
    file_path:
        Relative path to a file. Required for 'blame'; optional for 'log'.
    operation:
        'log'   – recent commits (optionally filtered to a file)
        'blame' – git blame for a line range (requires file_path)
        'diff'  – show recent changes diff for a file
    max_commits:
        How many commits to show (for 'log').
    start_line / end_line:
        Line range for 'blame'.
    """
    if not _git_available():
        return {"error": "git is not available on this system"}

    root = str(workspace.root)

    # Validate file path if provided
    resolved_path: str | None = None
    if file_path:
        try:
            rp = workspace.resolve(file_path)
            resolved_path = str(rp)
        except PathTraversalError as exc:
            return {"error": str(exc)}

    op = operation.lower()

    if op == "log":
        args = [
            "log",
            "--oneline",
            f"--max-count={max_commits}",
            "--no-merges",
            "--pretty=format:%h %ad %an: %s",
            "--date=short",
        ]
        if resolved_path:
            args += ["--", resolved_path]

        stdout, stderr, rc = _run_git(args, root)
        if rc != 0:
            return {"error": f"git log failed: {stderr.strip()}"}
        commits = [line for line in stdout.splitlines() if line.strip()]
        return {
            "operation": "log",
            "file": file_path,
            "commits": commits,
        }

    elif op == "blame":
        if not resolved_path:
            return {"error": "file_path is required for blame"}
        args = ["blame", "--porcelain"]
        if start_line and end_line:
            args += [f"-L{start_line},{end_line}"]
        args += [resolved_path]

        stdout, stderr, rc = _run_git(args, root)
        if rc != 0:
            return {"error": f"git blame failed: {stderr.strip()}"}

        # Parse porcelain output into readable lines
        blame_lines: list[dict[str, Any]] = []
        current: dict[str, Any] = {}
        lines = stdout.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line and not line.startswith("\t"):
                parts = line.split()
                if len(parts) >= 3 and len(parts[0]) == 40:
                    current = {"commit": parts[0][:8], "line": int(parts[2])}
            elif line.startswith("author "):
                current["author"] = line[7:]
            elif line.startswith("author-time "):
                current["timestamp"] = line[12:]
            elif line.startswith("summary "):
                current["summary"] = line[8:]
            elif line.startswith("\t"):
                current["content"] = line[1:]
                blame_lines.append(dict(current))
            i += 1

        return {
            "operation": "blame",
            "file": file_path,
            "blame": blame_lines,
        }

    elif op == "diff":
        if not resolved_path:
            return {"error": "file_path is required for diff"}
        # Show last commit's diff for this file
        args = ["log", "-1", "-p", "--", resolved_path]
        stdout, stderr, rc = _run_git(args, root)
        if rc != 0:
            return {"error": f"git diff failed: {stderr.strip()}"}
        # Limit output
        lines = stdout.splitlines()
        if len(lines) > 200:
            lines = lines[:200]
            lines.append("... (truncated)")
        return {
            "operation": "diff",
            "file": file_path,
            "diff": "\n".join(lines),
        }

    else:
        return {"error": f"Unknown operation '{operation}'. Use: log, blame, diff"}


# ── Tool declaration ──────────────────────────────────────────────────────────

TOOL_DEFINITION: dict[str, Any] = {
    "name": "git_history",
    "description": (
        "Inspect Git history to understand when and why code changed. "
        "Use 'log' to see recent commits for a file or the whole repo. "
        "Use 'blame' to see who last changed specific lines. "
        "Use 'diff' to see what changed in the last commit for a file. "
        "Only call this when git context is genuinely useful for the investigation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": (
                    "Relative path to the file. "
                    "Required for 'blame' and 'diff'; optional for 'log'."
                ),
            },
            "operation": {
                "type": "string",
                "description": (
                    "Operation to perform: 'log' (recent commits), "
                    "'blame' (line-level authorship), or 'diff' (last change)."
                ),
            },
            "max_commits": {
                "type": "string",
                "description": "Maximum number of commits to return for 'log'. Defaults to 10.",
            },
            "start_line": {
                "type": "string",
                "description": "Start line for blame (1-indexed).",
            },
            "end_line": {
                "type": "string",
                "description": "End line for blame (1-indexed).",
            },
        },
        "required": [],
    },
}
