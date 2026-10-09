"""
search_code tool — full-text / regex search across the repository.

Prefers ripgrep when available; falls back to pure-Python scanning.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from agent import config
from agent.repository.workspace import Workspace, PathTraversalError
from agent.repository.repository_scanner import (
    RepositoryScanner,
    SKIP_DIRS,
    SOURCE_EXTENSIONS,
)

_CONTEXT_LINES = 2  # lines before/after each match to include


# ── ripgrep ───────────────────────────────────────────────────────────────────

def _ripgrep_available() -> bool:
    return shutil.which("rg") is not None


def _run_ripgrep(
    query: str,
    base: Path,
    file_pattern: str | None,
    is_regex: bool,
) -> list[dict[str, Any]]:
    cmd = [
        "rg",
        "--json",
        "--context", str(_CONTEXT_LINES),
        "--max-count", "5",          # max 5 matches per file
        "--max-filesize", f"{config.MAX_FILE_SIZE_BYTES}",
    ]
    if not is_regex:
        cmd.append("--fixed-strings")
    if file_pattern:
        cmd += ["--glob", file_pattern]
    cmd += [query, str(base)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return _parse_ripgrep_json(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def _parse_ripgrep_json(output: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    current_match: dict[str, Any] | None = None
    context_before: list[str] = []
    context_after: list[str] = []

    for raw_line in output.splitlines():
        if not raw_line.strip():
            continue
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        msg_type = obj.get("type")

        if msg_type == "match":
            data = obj["data"]
            file_path = data["path"]["text"]
            for sub in data["submatches"]:
                line_no = data["line_number"]
                line_text = data["lines"]["text"].rstrip("\n")
                matches.append(
                    {
                        "file": file_path,
                        "line": line_no,
                        "match": line_text,
                        "context_before": list(context_before),
                        "context_after": [],
                    }
                )
                context_before = []
        elif msg_type == "context":
            data = obj["data"]
            line_text = data["lines"]["text"].rstrip("\n")
            if matches and data["line_number"] > matches[-1]["line"]:
                matches[-1]["context_after"].append(line_text)
            else:
                context_before.append(line_text)
                if len(context_before) > _CONTEXT_LINES:
                    context_before.pop(0)

    return matches


# ── Python fallback ───────────────────────────────────────────────────────────

def _python_search(
    query: str,
    base: Path,
    file_pattern: str | None,
    is_regex: bool,
    workspace: Workspace,
) -> list[dict[str, Any]]:
    scanner = RepositoryScanner(workspace)
    pattern = re.compile(query if is_regex else re.escape(query), re.IGNORECASE)
    results: list[dict[str, Any]] = []

    # Build glob filter if provided
    glob_regex: re.Pattern | None = None
    if file_pattern:
        try:
            import fnmatch
            glob_pat = fnmatch.translate(file_pattern)
            glob_regex = re.compile(glob_pat, re.IGNORECASE)
        except Exception:
            glob_regex = None

    for fpath in scanner.iter_source_files(base if base != workspace.root else None):
        if glob_regex and not glob_regex.search(fpath.name):
            continue
        if scanner.is_binary(fpath):
            continue
        if scanner.file_size(fpath) > config.MAX_FILE_SIZE_BYTES:
            continue

        try:
            lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue

        file_matches = 0
        for i, line in enumerate(lines):
            if file_matches >= 5:
                break
            if pattern.search(line):
                before = lines[max(0, i - _CONTEXT_LINES): i]
                after = lines[i + 1: i + 1 + _CONTEXT_LINES]
                results.append(
                    {
                        "file": str(fpath),
                        "line": i + 1,
                        "match": line,
                        "context_before": before,
                        "context_after": after,
                    }
                )
                file_matches += 1

    return results


# ── public API ────────────────────────────────────────────────────────────────

MAX_TOTAL_RESULTS = 50


def search_code(
    workspace: Workspace,
    query: Any,
    path: str = ".",
    file_pattern: str | None = None,
    is_regex: bool = False,
) -> dict[str, Any]:
    """
    Search for *query* across source files in the repository.

    Returns
    -------
    dict with keys:
        "query"      – the original query
        "matches"    – list of match dicts (file, line, match, context_*)
        "truncated"  – True if results were capped
        "error"      – present only on failure
    """
    # 1. Normalize and validate query
    if query is None:
        return {"error": "query must not be empty"}
    if isinstance(query, (int, float, bool)):
        query = str(query)
    elif not isinstance(query, str):
        return {"error": f"Invalid query: expected string or number, got {type(query).__name__}"}

    query = query.strip()
    if not query:
        return {"error": "query must not be empty"}

    # 2. Normalize file_pattern
    if file_pattern is not None:
        if not isinstance(file_pattern, str):
            file_pattern = str(file_pattern)
        file_pattern = file_pattern.strip() or None

    # 3. Normalize and validate regex
    if isinstance(is_regex, str):
        is_regex = is_regex.strip().lower() in ("true", "1", "yes")
    else:
        is_regex = bool(is_regex)

    if is_regex:
        try:
            re.compile(query)
        except re.error as exc:
            return {"error": f"Invalid regular expression '{query}': {exc}"}

    # 4. Resolve path
    try:
        base = workspace.resolve(path or ".")
    except PathTraversalError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": f"Invalid search path: {exc}"}

    # 5. Execute search
    try:
        if _ripgrep_available():
            matches = _run_ripgrep(query, base, file_pattern, is_regex)
        else:
            matches = _python_search(query, base, file_pattern, is_regex, workspace)
    except Exception as exc:
        return {"error": f"Search execution failed: {exc}", "query": query, "matches": []}

    truncated = len(matches) > MAX_TOTAL_RESULTS
    matches = matches[:MAX_TOTAL_RESULTS]

    # Make file paths relative to repo root for cleaner output
    for m in matches:
        m["file"] = workspace.relative(m["file"])

    return {
        "query": query,
        "total_matches": len(matches),
        "truncated": truncated,
        "matches": matches,
    }


# ── Tool declaration ──────────────────────────────────────────────────────────

TOOL_DEFINITION: dict[str, Any] = {
    "name": "search_code",
    "description": (
        "Search for a keyword, text snippet, or regex pattern across source files "
        "in the repository. Returns file name, line number, the matching line, and "
        "a small surrounding context. Use this to locate relevant classes, methods, "
        "or patterns before reading the full file. Prefer specific terms over broad ones."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search term, keyword, or regex pattern to find.",
            },
            "path": {
                "type": "string",
                "description": (
                    "Sub-directory within the repository to restrict the search to. "
                    "Use '.' to search the entire repository."
                ),
            },
            "file_pattern": {
                "type": "string",
                "description": (
                    "Optional glob pattern to filter files, e.g. '*.java' or '*.py'."
                ),
            },
            "is_regex": {
                "type": "string",
                "description": (
                    "Set to 'true' if query is a regular expression. "
                    "Defaults to 'false' (literal search)."
                ),
            },
        },
        "required": ["query"],
    },
}
