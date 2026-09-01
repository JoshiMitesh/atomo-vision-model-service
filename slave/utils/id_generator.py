"""Thread-safe monotonically-increasing frame id generator.

Each job/worker owns its own instance — frame ids must never be shared
between jobs (see CLAUDE.md §20 Process Isolation).
"""

from __future__ import annotations

import threading


class FrameIdGenerator:
    def __init__(self, start: int = 0) -> None:
        self._value = start
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            self._value += 1
            return self._value
