"""Live view of what is actually running, for the device health message.

jobs.yaml is the *desired* job list (CLAUDE.md §3). This tracks the
*actual* outcome of acting on it — which model workers were built for
each job, and which assignments failed to build — so the health
message can report running_jobs / failed_assignments / per-camera
status instead of just echoing the configuration.

Written to by ModelManager/JobManager (job thread), read by the device
agent's telemetry loop (its own thread) — hence the lock.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Tuple

from slave.processes.worker.worker_manager import WorkerManager

MAX_ERROR_LENGTH = 300


class RuntimeStatus:
    def __init__(self, worker_manager: WorkerManager) -> None:
        self._worker_manager = worker_manager
        self._lock = threading.Lock()
        self._built: Dict[str, List[str]] = {}
        self._failed: Dict[str, List[dict]] = {}

    def job_built(self, job_id: str, model_types: List[str]) -> None:
        with self._lock:
            self._built[job_id] = list(model_types)

    def job_removed(self, job_id: str) -> None:
        with self._lock:
            self._built.pop(job_id, None)
            self._failed.pop(job_id, None)

    def reset_failures(self, job_id: str) -> None:
        with self._lock:
            self._failed.pop(job_id, None)

    def record_failure(self, job_id: str, model_type: str, model_name: str, error: str) -> None:
        entry = {
            "job_id": job_id,
            "model_type": model_type,
            "model_name": model_name,
            "error": error[:MAX_ERROR_LENGTH],
            "ts": int(time.time() * 1000),
        }
        with self._lock:
            self._failed.setdefault(job_id, []).append(entry)

    def snapshot(self) -> Tuple[Dict[str, List[str]], List[dict]]:
        """(running_jobs, failed_assignments). A job only counts as
        running while WorkerManager still owns its workers.
        """
        with self._lock:
            built = {job_id: list(types) for job_id, types in self._built.items()}
            failed = [dict(entry) for entries in self._failed.values() for entry in entries]
        running = {
            job_id: types
            for job_id, types in built.items()
            if types and self._worker_manager.is_running(job_id)
        }
        return running, failed

    def camera_status(self, job_id: str) -> str:
        """'online' / 'offline' — read from the live RTSP connection."""
        return self._worker_manager.camera_status(job_id)
