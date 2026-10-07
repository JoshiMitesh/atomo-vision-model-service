"""Executes device-level commands from the Master.

Distinct from job start/stop — those flow through processes/subscriber/
and config/jobs.yaml (CLAUDE.md §5-6). This handles device
administration: status queries and (eventually) model preload/reboot,
addressed directly at this device rather than at a job.

Command shape:   {"op": "get_status", "params": {...}}
Response shape:  {"status": "ok" | "error", "op": "...", "detail": ...}
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from slave.logging.logger import get_logger
from slave.processes.device.runtime_status import RuntimeStatus
from slave.processes.device.telemetry import sample_telemetry, uptime_seconds

log = get_logger(__name__)


def execute_command(cmd: Dict[str, Any], jobs_yaml_path: Optional[str] = None, device_name: str = "unknown-device", ip: str = "", runtime: Optional[RuntimeStatus] = None) -> Dict[str, Any]:
    op = cmd.get("op")
    params = cmd.get("params", {})

    if op == "get_status":
        telemetry = sample_telemetry(jobs_yaml_path, device_name, ip, runtime)
        return {
            "status": "ok",
            "op": op,
            "detail": {
                "device_name": telemetry.get("device_name"),
                "ip": telemetry.get("ip"),
                "cpu": telemetry.get("cpu"),
                "disk": telemetry.get("disk"),
                "running_cameras": telemetry.get("running_cameras"),
                "loaded_models": telemetry.get("loaded_models"),
                "cameras": telemetry.get("cameras"),
                "running_jobs": telemetry.get("running_jobs"),
                "failed_assignments": telemetry.get("failed_assignments"),
                "uptime_s": uptime_seconds(),
            },
        }

    if op == "load_model":
        # Placeholder. Today, model loading happens implicitly per-job
        # via config/jobs.yaml + job_watcher.py (CLAUDE.md §3-4). Wire
        # this up only if the Master needs to pre-load/warm a model
        # independent of an active job.
        model = params.get("model")
        return {"status": "ok", "op": op, "detail": f"{model} load acknowledged (not yet implemented)"}

    if op == "unload_model":
        model = params.get("model")
        return {"status": "ok", "op": op, "detail": f"{model} unload acknowledged (not yet implemented)"}

    if op == "reboot":
        # Deliberately NOT wired to a real reboot syscall — a remote
        # command that actually reboots hardware needs an explicit,
        # separate decision before going live. Acknowledge only.
        log.warning("reboot command received but not wired to a real reboot — acknowledging only")
        return {"status": "ok", "op": op, "detail": "reboot acknowledged (not yet implemented)"}

    log.warning("unknown device command: %s", op)
    return {"status": "error", "op": op, "detail": "unknown command"}
