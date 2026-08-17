"""Small filesystem helpers shared across processes."""

from __future__ import annotations

import os
import tempfile


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def atomic_write_text(path: str, content: str, encoding: str = "utf-8") -> None:
    """Write ``content`` to ``path`` without ever leaving a half-written
    file behind (write to a temp file in the same directory, then
    ``os.replace``, which is atomic on POSIX and Windows).
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    ensure_dir(directory)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(content)
        os.replace(tmp_path, path)
    except Exception:
        safe_remove(tmp_path)
        raise


def safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
