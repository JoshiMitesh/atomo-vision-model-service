"""Time helpers shared across processes."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

# Fixed UTC+5:30 offset, not the host machine's own local timezone —
# every DetectionResult.occurred_at should read as Indian time
# regardless of what timezone the board/dev machine's system clock is
# actually set to.
IST = timezone(timedelta(hours=5, minutes=30))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ist_now_iso() -> str:
    """Current time as an ISO 8601 string in IST (UTC+5:30) — used for
    DetectionResult.occurred_at instead of utc_now_iso() so the
    timestamp Master/consumers see is Indian wall-clock time, not UTC.
    """
    return datetime.now(IST).isoformat()


def monotonic() -> float:
    return time.monotonic()


def epoch_seconds() -> float:
    return time.time()
