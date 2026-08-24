"""Device-level hardware telemetry (CPU/disk/uptime).

Shared by telemetry_publisher.py, heartbeat_publisher.py, and
command_handler.py's get_status response — one sampling function, not
duplicated three times.
"""

from __future__ import annotations

import getpass
import socket
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


def _detect_local_ip() -> Optional[str]:
    """Best-effort LAN IP of this machine — the address other hosts
    (Master, NATS) would actually use to reach it, not '127.0.0.1'.

    Uses the UDP-connect trick (no packet is actually sent) to ask the
    OS which local interface it would route through.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return None


def _detect_slave_name() -> Optional[str]:
    """OS username this process runs as — a human-readable label for
    the device, distinct from the configured slave_id.
    """
    try:
        return getpass.getuser()
    except Exception:
        return None


# Cached once at import time — these don't change while the process runs.
_LOCAL_IP = _detect_local_ip()
_SLAVE_NAME = _detect_slave_name()


def _detect_temp_c() -> Optional[float]:
    """First available hardware temperature sensor reading, in Celsius.

    Falls back to a mock value when no sensor is exposed (e.g. inside a
    container, or on a dev laptop without a supported chip) — matches
    the reference slave1.py behavior of reporting *some* temp rather
    than leaving the field permanently null.
    """
    if not hasattr(psutil, "sensors_temperatures"):
        return 45.0
    try:
        temps = psutil.sensors_temperatures()
    except Exception:
        return 45.0
    if not temps:
        return 45.0
    first_sensor = next(iter(temps.values()), None)
    if not first_sensor:
        return 45.0
    return first_sensor[0].current


def sample_telemetry(slave_id: Optional[str] = None) -> dict:
    payload: dict = {
        "slave_id": slave_id,
        "slave_name": _SLAVE_NAME,
        "ip": _LOCAL_IP,
        "ts": int(time.time() * 1000),
    }
    if psutil is None:
        return payload
    payload["cpu"] = psutil.cpu_percent(interval=None)
    payload["temp"] = _detect_temp_c()
    payload["disk"] = psutil.disk_usage("/").percent
    payload["ram"] = psutil.virtual_memory().percent
    payload["npu"] = 0.0  # placeholder — real NPU utilization depends on A311D SDK, not wired up yet
    return payload


def uptime_seconds() -> Optional[float]:
    if psutil is None:
        return None
    return time.time() - psutil.boot_time()
