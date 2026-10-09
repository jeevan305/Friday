"""
Repository workspace — enforces path confinement.

All tool file access must go through Workspace.resolve() to prevent
path traversal outside the configured repository root.
"""

from __future__ import annotations

import os
from pathlib import Path


class PathTraversalError(Exception):
    """Raised when a path would escape the repository root."""


class Workspace:
    """Safe, confined access to a repository on disk."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        if not self._root.exists():
            raise FileNotFoundError(
                f"Repository path does not exist: {self._root}"
            )
        if not self._root.is_dir():
            raise NotADirectoryError(
                f"Repository path is not a directory: {self._root}"
            )

    @property
    def root(self) -> Path:
        return self._root

    def resolve(self, relative_or_absolute: str | Path) -> Path:
        """
        Resolve *relative_or_absolute* relative to the repository root.

        Raises PathTraversalError if the result escapes the root.
        """
        p = Path(relative_or_absolute)
        if p.is_absolute():
            resolved = p.resolve()
        else:
            resolved = (self._root / p).resolve()

        try:
            resolved.relative_to(self._root)
        except ValueError:
            raise PathTraversalError(
                f"Path '{relative_or_absolute}' escapes the repository root "
                f"'{self._root}'. Access denied."
            )
        return resolved

    def relative(self, absolute: str | Path) -> str:
        """Return a path relative to the repo root (for display purposes)."""
        try:
            return str(Path(absolute).resolve().relative_to(self._root))
        except ValueError:
            return str(absolute)

    def exists(self, path: str | Path) -> bool:
        try:
            return self.resolve(path).exists()
        except PathTraversalError:
            return False

    def __repr__(self) -> str:
        return f"Workspace(root={self._root})"
