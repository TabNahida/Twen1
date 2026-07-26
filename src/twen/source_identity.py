"""Deterministic identity for the Python source that implements Twen."""

from __future__ import annotations

import hashlib
from pathlib import Path

SOURCE_TREE_HASH_SCHEMA = b"twen-python-source-tree-v1\0"


def twen_source_tree_sha256(root: str | Path | None = None) -> str:
    """Hash every ``src/twen/**/*.py`` path and byte exactly once.

    Length-prefixing both the POSIX relative path and content makes file
    boundaries unambiguous. Sorting paths makes the result independent of
    directory iteration order, while including paths detects moves/renames in
    addition to content changes. Non-Python runtime caches are intentionally
    excluded.
    """

    source_root = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parent
    )
    files = sorted(
        path
        for path in source_root.rglob("*.py")
        if path.is_file()
    )
    if not files:
        raise ValueError(f"Twen source tree contains no Python files: {source_root}")

    digest = hashlib.sha256()
    digest.update(SOURCE_TREE_HASH_SCHEMA)
    for path in files:
        relative = path.relative_to(source_root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


__all__ = ["SOURCE_TREE_HASH_SCHEMA", "twen_source_tree_sha256"]
