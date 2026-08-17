"""Device-level hardware telemetry (CPU/disk/uptime).

Shared by telemetry_publisher.py, heartbeat_publisher.py, and
command_handler.py's get_status response — one sampling function, not
duplicated three times.
"""

from __future__ import annotations

import time
from typing import Optional

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore[assignment]

from slave.logging.logger import get_logger

log = get_logger(__name__)

if psutil is None:
    log.warning(
        "psutil not installed — telemetry/heartbeat payloads will only "
        "contain 'ts' (no cpu/disk). Run: pip install psutil"
    )
else:
    # Prime the internal baseline so the first real reading reflects
    # actual CPU load instead of 0.0.
    psutil.cpu_percent(interval=None)


def sample_telemetry() -> dict:
    if psutil is None:
        return {"ts": int(time.time() * 1000)}
    return {
        "cpu": psutil.cpu_percent(interval=None),
        "temp": None,  # system-specific hardware temp sensor, not wired up yet
        "disk": psutil.disk_usage("/").percent,
        "ts": int(time.time() * 1000),
    }


def uptime_seconds() -> Optional[float]:
    if psutil is None:
        return None
    return time.time() - psutil.boot_time()
