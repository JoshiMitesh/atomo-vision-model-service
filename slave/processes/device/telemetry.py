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
from typing import Optional

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore[assignment]

from slave.logging.logger import get_logger
from slave.processes.device.runtime_status import RuntimeStatus
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


def _job_stats(jobs_yaml_path: Optional[str], runtime: Optional[RuntimeStatus] = None) -> dict:
    """running_cameras / loaded_models / cameras, read live from
    config/jobs.yaml (CLAUDE.md §3: jobs.yaml is runtime state), plus
    running_jobs / failed_assignments / per-camera status from
    `runtime` (what actually got built and is connected).

    loaded_models lists every assigned model, not just the primary
    one. Never raises — a bad/missing jobs.yaml must not break
    telemetry.
    """
    stats: dict = {
        "running_cameras": 0,
        "loaded_models": [],
        "cameras": [],
        "running_jobs": {},
        "failed_assignments": [],
    }
    if runtime is not None:
        stats["running_jobs"], stats["failed_assignments"] = runtime.snapshot()

    if not jobs_yaml_path:
        return stats
    try:
        jobs = job_store.load_jobs(jobs_yaml_path)
    except Exception:
        log.exception("failed to read jobs.yaml for telemetry job stats")
        return stats

    stats["running_cameras"] = len(jobs)
    stats["loaded_models"] = sorted(
        {assignment.model_name for job in jobs for assignment in job.model_assignments}
    )
    stats["cameras"] = [
        {
            "camera_id": job.camera_id,
            "camera_name": job.camera_name,
            "status": runtime.camera_status(job.job_id) if runtime is not None else "offline",
            "latitude": job.latitude,
            "longitude": job.longitude,
            "location": job.location,
        }
        for job in jobs
    ]
    return stats


def sample_telemetry(
    jobs_yaml_path: Optional[str],
    device_name: str,
    ip: str,
    runtime: Optional[RuntimeStatus] = None,
) -> dict:
    payload: dict = {
        "device_name": device_name,
        "ip": ip or None,
        "ts": int(time.time() * 1000),
    }
    payload.update(_job_stats(jobs_yaml_path, runtime))

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
