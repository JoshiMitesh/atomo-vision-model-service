"""Time helpers shared across processes."""

from __future__ import annotations

import time
from datetime import datetime, timezone


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def monotonic() -> float:
    return time.monotonic()


def epoch_seconds() -> float:
    return time.time()
