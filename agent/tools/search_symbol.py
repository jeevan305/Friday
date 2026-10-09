"""
search_symbol tool — locate classes, methods, functions, and other named symbols.

Supports Java and Python with regex-based heuristics.
Designed to be extended for additional languages later.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agent import config
from agent.repository.workspace import Workspace, PathTraversalError
from agent.repository.repository_scanner import RepositoryScanner

# ── Language patterns ─────────────────────────────────────────────────────────

_JAVA_PATTERNS: list[tuple[str, str]] = [
    # class / interface / enum / record
    (
        r"(?:public|protected|private|abstract|final|static)?\s*"
        r"(?:class|interface|enum|record)\s+(\w+)",
        "class",
    ),
    # method (simplified — catches most cases)
    (
        r"(?:public|protected|private|static|final|abstract|synchronized|native|default)[\w\s<>\[\],?]*"
        r"\s+(\w+)\s*\(",
        "method",
    ),
    # annotation type
    (r"@interface\s+(\w+)", "annotation"),
]

_PYTHON_PATTERNS: list[tuple[str, str]] = [
    (r"^class\s+(\w+)", "class"),
    (r"^def\s+(\w+)", "function"),
    (r"^\s+def\s+(\w+)", "method"),
]

_LANGUAGE_MAP: dict[str, list[tuple[str, str]]] = {
    ".java": _JAVA_PATTERNS,
    ".py": _PYTHON_PATTERNS,
    ".kt": _JAVA_PATTERNS,  # Kotlin is close enough for basic patterns
}


def _scan_file_for_symbol(
    fpath: Path,
    symbol_pattern: re.Pattern,
    workspace: Workspace,
) -> list[dict[str, Any]]:
    ext = fpath.suffix.lower()
    lang_patterns = _LANGUAGE_MAP.get(ext)
    if not lang_patterns:
        return []

    try:
        lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []

    results: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        for lang_re_str, symbol_type in lang_patterns:
            m = re.search(lang_re_str, line)
            if m:
                name = m.group(1)
                if symbol_pattern.search(name):
                    # Find end of block (simple heuristic: next same-or-higher-level declaration)
                    start_line = i + 1
                    end_line = _find_block_end(lines, i, ext)
                    results.append(
                        {
                            "file": workspace.relative(fpath),
                            "symbol": name,
                            "symbol_type": symbol_type,
                            "start_line": start_line,
                            "end_line": end_line,
                        }
                    )
        i += 1
    return results


def _find_block_end(lines: list[str], start: int, ext: str) -> int:
    """
    Heuristic: scan forward to estimate the end line of the block
    starting at *start*.
    """
    if ext == ".py":
        # Python: end when indent returns to same or lower level
        try:
            base_indent = len(lines[start]) - len(lines[start].lstrip())
            for j in range(start + 1, min(start + 200, len(lines))):
                stripped = lines[j].rstrip()
                if not stripped or stripped.startswith("#"):
                    continue
                indent = len(lines[j]) - len(lines[j].lstrip())
                if indent <= base_indent and stripped:
                    return j  # exclusive
        except IndexError:
            pass
        return min(start + 50, len(lines))
    else:
        # Java/Kotlin: count braces
        depth = 0
        for j in range(start, min(start + 300, len(lines))):
            depth += lines[j].count("{") - lines[j].count("}")
            if j > start and depth <= 0:
                return j + 1
        return min(start + 100, len(lines))


def search_symbol(
    workspace: Workspace,
    symbol: str,
    path: str = ".",
    file_pattern: str | None = None,
) -> dict[str, Any]:
    """
    Search for a named symbol (class, method, function) in the repository.

    Returns
    -------
    dict with keys:
        "symbol"  – the original symbol query
        "results" – list of symbol dicts (file, symbol, symbol_type, start_line, end_line)
        "error"   – present only on failure
    """
    if not symbol or not symbol.strip():
        return {"error": "symbol must not be empty"}

    try:
        base = workspace.resolve(path)
    except PathTraversalError as exc:
        return {"error": str(exc)}

    try:
        sym_pattern = re.compile(symbol, re.IGNORECASE)
    except re.error as exc:
        return {"error": f"Invalid symbol pattern: {exc}"}

    scanner = RepositoryScanner(workspace)
    all_results: list[dict[str, Any]] = []

    # Build glob filter
    glob_re: re.Pattern | None = None
    if file_pattern:
        gp = file_pattern.replace(".", r"\.").replace("*", ".*").replace("?", ".")
        glob_re = re.compile(gp, re.IGNORECASE)

    for fpath in scanner.iter_source_files(base if base != workspace.root else None):
        if glob_re and not glob_re.search(fpath.name):
            continue
        if scanner.is_binary(fpath):
            continue
        if scanner.file_size(fpath) > config.MAX_FILE_SIZE_BYTES:
            continue
        all_results.extend(_scan_file_for_symbol(fpath, sym_pattern, workspace))

    return {
        "symbol": symbol,
        "total_results": len(all_results),
        "results": all_results,
    }


# ── Tool declaration ──────────────────────────────────────────────────────────

TOOL_DEFINITION: dict[str, Any] = {
    "name": "search_symbol",
    "description": (
        "Find a named symbol — class, method, function, or annotation — "
        "anywhere in the repository. Supports Java and Python. "
        "Returns the file, symbol name, type, and line range. "
        "Use this to jump directly to a class or method definition "
        "when you know (or suspect) its name."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": (
                    "Symbol name or pattern to search for. "
                    "Examples: 'OrderService', 'createOrder', 'validateCustomer'."
                ),
            },
            "path": {
                "type": "string",
                "description": (
                    "Sub-directory to restrict the search to. "
                    "Use '.' to search the whole repository."
                ),
            },
            "file_pattern": {
                "type": "string",
                "description": "Optional glob to restrict file types, e.g. '*.java'.",
            },
        },
        "required": ["symbol"],
    },
}
