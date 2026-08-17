"""Factory for the shared 10 MB rotating file handler.

Kept in its own module so ``log_manager.py`` stays focused on
composition rather than handler mechanics.
"""

from __future__ import annotations

import logging as stdlib_logging
import logging.handlers as stdlib_handlers
import os

DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5


def build_rotating_file_handler(
    log_path: str,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> stdlib_logging.Handler:
    """Build a RotatingFileHandler that rolls over once the active file
    reaches ``max_bytes`` (default 10 MB), keeping ``backup_count`` old
    files around.
    """
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    handler = stdlib_handlers.RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    return handler
