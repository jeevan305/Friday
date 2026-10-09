"""
list_directory tool — explore repository structure.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agent.repository.workspace import Workspace, PathTraversalError
from agent.repository.repository_scanner import SKIP_DIRS


def list_directory(workspace: Workspace, path: str = ".") -> dict[str, Any]:
    """
    List the contents of *path* within the repository.

    Returns
    -------
    dict with keys:
        "path"        – the path that was listed (relative to repo root)
        "directories" – sorted list of subdirectory names
        "files"       – sorted list of file names
        "error"       – present only if something went wrong
    """
    try:
        resolved = workspace.resolve(path)
    except PathTraversalError as exc:
        return {"error": str(exc)}

    if not resolved.exists():
        return {"error": f"Path does not exist: {path}"}
    if not resolved.is_dir():
        return {"error": f"Path is not a directory: {path}"}

    dirs: list[str] = []
    files: list[str] = []

    try:
        for entry in sorted(resolved.iterdir()):
            name = entry.name
            if name in SKIP_DIRS:
                continue
            if entry.is_dir():
                dirs.append(name)
            elif entry.is_file():
                files.append(name)
    except PermissionError as exc:
        return {"error": f"Permission denied: {exc}"}

    return {
        "path": workspace.relative(resolved),
        "directories": dirs,
        "files": files,
    }


# ── Tool declaration (provider-agnostic) ─────────────────────────────────────

TOOL_DEFINITION: dict[str, Any] = {
    "name": "list_directory",
    "description": (
        "List the files and sub-directories at a path inside the repository. "
        "Use this to explore the project structure before deciding which files to read. "
        "Pass '.' to list the root of the repository."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Relative path within the repository to list. "
                    "Use '.' for the root. Example: 'src/main/java/com/example'"
                ),
            },
        },
        "required": [],
    },
}
