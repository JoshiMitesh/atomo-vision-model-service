"""Device-level hardware telemetry (CPU/disk/uptime) + live job stats.

Shared by telemetry_publisher.py, heartbeat_publisher.py, and
command_handler.py's get_status response — one sampling function, not
duplicated three times.

device_name/ip (config.DeviceConfig) are passed in by the caller and
included in the payload — the Master requires them in the health
message body even though the subject is already per-device.
"""

from __future__ import annotations

import time
from typing import List, Optional

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore[assignment]

from slave.logging.logger import get_logger
from slave.storage.yaml import job_store

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


def _job_stats(jobs_yaml_path: Optional[str]) -> dict:
    """running_cameras / loaded_models, read live from config/jobs.yaml
    (CLAUDE.md §3: jobs.yaml is runtime state) — not from any
    in-memory worker registry, so this reflects the desired job list
    the instant it's published, same as job_watcher.py will pick up.

    Never raises — a bad/missing jobs.yaml must not break telemetry.
    """
    if not jobs_yaml_path:
        return {"running_cameras": 0, "loaded_models": []}
    try:
        jobs = job_store.load_jobs(jobs_yaml_path)
    except Exception:
        log.exception("failed to read jobs.yaml for telemetry job stats")
        return {"running_cameras": 0, "loaded_models": []}

    loaded_models: List[str] = sorted({job.model_name for job in jobs})
    return {"running_cameras": len(jobs), "loaded_models": loaded_models}


def sample_telemetry(jobs_yaml_path: Optional[str], device_name: str, ip: str) -> dict:
    payload: dict = {
        "device_name": device_name,
        "ip": ip or None,
        "ts": int(time.time() * 1000),
    }
    payload.update(_job_stats(jobs_yaml_path))

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
