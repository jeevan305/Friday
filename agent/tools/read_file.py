"""
read_file tool — read a specific range of lines from a source file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent import config
from agent.repository.workspace import Workspace, PathTraversalError
from agent.repository.repository_scanner import RepositoryScanner


def read_file(
    workspace: Workspace,
    file_path: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> dict[str, Any]:
    """
    Read lines *start_line* through *end_line* (1-indexed, inclusive) from *file_path*.

    If *end_line* is None, reads up to MAX_READ_LINES lines from *start_line*.

    Returns
    -------
    dict with keys:
        "file"          – relative path
        "start_line"    – first line returned
        "end_line"      – last line returned
        "total_lines"   – total lines in file
        "content"       – formatted string  "NNN | <source line>"
        "truncated"     – True if the file was larger than MAX_READ_LINES
        "error"         – present only on failure
    """
    try:
        resolved = workspace.resolve(file_path)
    except PathTraversalError as exc:
        return {"error": str(exc)}

    if not resolved.exists():
        return {"error": f"File not found: {file_path}"}
    if not resolved.is_file():
        return {"error": f"Not a file: {file_path}"}

    scanner = RepositoryScanner(workspace)
    if scanner.is_binary(resolved):
        return {"error": f"Binary file, cannot read as text: {file_path}"}

    file_size = scanner.file_size(resolved)
    if file_size > config.MAX_FILE_SIZE_BYTES:
        return {
            "error": (
                f"File is too large to read ({file_size} bytes). "
                f"Use search_code to locate the relevant portion first."
            )
        }

    try:
        all_lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return {"error": f"Cannot read file: {exc}"}

    total_lines = len(all_lines)

    # Normalise indices
    start = max(1, start_line)
    if end_line is None:
        end = min(total_lines, start + config.MAX_READ_LINES - 1)
    else:
        end = min(total_lines, end_line)

    if start > total_lines:
        return {
            "error": (
                f"start_line {start} exceeds file length ({total_lines} lines)."
            )
        }

    selected = all_lines[start - 1: end]  # 0-indexed slice

    # Format with line numbers
    width = len(str(end))
    formatted_lines = [
        f"{start + i:>{width}} | {line}"
        for i, line in enumerate(selected)
    ]
    content = "\n".join(formatted_lines)

    truncated = (end - start + 1) >= config.MAX_READ_LINES and end < total_lines

    return {
        "file": workspace.relative(resolved),
        "start_line": start,
        "end_line": end,
        "total_lines": total_lines,
        "content": content,
        "truncated": truncated,
    }


# ── Tool declaration ──────────────────────────────────────────────────────────

TOOL_DEFINITION: dict[str, Any] = {
    "name": "read_file",
    "description": (
        "Read a specific range of lines from a source file in the repository. "
        "Always specify a focused range rather than reading an entire large file. "
        "Line numbers are shown alongside the code. Use search_code first to "
        "discover which lines are relevant, then use read_file to inspect them in detail."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Relative path to the file within the repository.",
            },
            "start_line": {
                "type": "string",
                "description": "First line to read (1-indexed). Defaults to 1.",
            },
            "end_line": {
                "type": "string",
                "description": (
                    "Last line to read (1-indexed, inclusive). "
                    "If omitted, reads up to 300 lines from start_line."
                ),
            },
        },
        "required": ["file_path"],
    },
}
