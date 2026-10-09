"""
Repository scanner — lightweight metadata about a repository.

Used by tools to quickly enumerate files without loading content.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from agent.repository.workspace import Workspace

# Extensions considered source code for investigation
SOURCE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".java", ".py", ".kt", ".groovy", ".scala",
        ".js", ".ts", ".jsx", ".tsx",
        ".go", ".rs", ".c", ".cpp", ".h", ".cs",
        ".rb", ".php", ".swift",
        ".xml", ".yaml", ".yml", ".json", ".properties",
        ".sql", ".sh", ".md",
    }
)

# Directories to always skip
SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git", ".svn", ".hg",
        "node_modules", "__pycache__", ".pytest_cache",
        "target", "build", "dist", "out", ".idea", ".vscode",
        "venv", ".venv", "env", ".env",
        "coverage", ".coverage",
    }
)


class RepositoryScanner:
    """Provides metadata about files in a workspace without loading content."""

    def __init__(self, workspace: Workspace) -> None:
        self._ws = workspace

    def iter_source_files(
        self,
        subpath: str | Path | None = None,
    ) -> Iterator[Path]:
        """Yield all source files under *subpath* (default: repo root)."""
        base = self._ws.resolve(subpath) if subpath else self._ws.root
        for dirpath, dirnames, filenames in os.walk(base):
            # Prune skip dirs in-place so os.walk won't recurse into them
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            dp = Path(dirpath)
            for fname in filenames:
                fpath = dp / fname
                if fpath.suffix.lower() in SOURCE_EXTENSIONS:
                    yield fpath

    def is_binary(self, path: Path) -> bool:
        """Heuristic binary detection (read first 8 KB)."""
        try:
            with open(path, "rb") as fh:
                chunk = fh.read(8192)
            return b"\x00" in chunk
        except OSError:
            return False

    def file_size(self, path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0
