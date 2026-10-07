"""Live view of what is actually running, for the device health message.

jobs.yaml is the *desired* job list (CLAUDE.md §3). This tracks the
*actual* outcome of acting on it — which model workers were built for
each job, which failed to build or start, and which have died since —
so the health message can report running_jobs / failed_assignments /
per-camera status instead of just echoing the configuration.

Written to by ModelManager/JobManager (job thread), read by the device
agent's telemetry loop (its own thread) — hence the lock.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from slave.processes.worker.worker_manager import WorkerManager

MAX_ERROR_LENGTH = 300


@dataclass
class _Entry:
    worker: object
    model_type: str
    model_name: str
    started: bool = False
    start_error: Optional[str] = None
    died_at: Optional[int] = None  # ms epoch, set when a started worker is seen dead


class RuntimeStatus:
    def __init__(self, worker_manager: WorkerManager) -> None:
        self._worker_manager = worker_manager
        self._lock = threading.Lock()
        self._entries: Dict[str, List[_Entry]] = {}
        self._failed: Dict[str, List[dict]] = {}  # assignments that never produced a worker

    def job_built(self, job_id: str, built: List[Tuple[object, str, str]]) -> None:
        """built: [(worker, model_type, model_name), ...]"""
        with self._lock:
            self._entries[job_id] = [_Entry(w, t, n) for w, t, n in built]

    def job_started(self, job_id: str, start_failures: List[Tuple[object, Exception]]) -> None:
        failed = {id(w): exc for w, exc in start_failures}
        with self._lock:
            for entry in self._entries.get(job_id, []):
                if id(entry.worker) in failed:
                    exc = failed[id(entry.worker)]
                    entry.start_error = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_LENGTH]
                else:
                    entry.started = True

    def replace_worker(self, job_id: str, old: object, new: object, error: Optional[str] = None) -> None:
        with self._lock:
            for entry in self._entries.get(job_id, []):
                if entry.worker is old:
                    entry.worker = new
                    entry.started = error is None
                    entry.start_error = error[:MAX_ERROR_LENGTH] if error else None
                    entry.died_at = None

    def job_removed(self, job_id: str) -> None:
        with self._lock:
            self._entries.pop(job_id, None)
            self._failed.pop(job_id, None)

    def reset_failures(self, job_id: str) -> None:
        with self._lock:
            self._failed.pop(job_id, None)

    def record_failure(self, job_id: str, model_type: str, model_name: str, error: str) -> None:
        with self._lock:
            self._failed.setdefault(job_id, []).append(self._failure(job_id, model_type, model_name, error))

    @staticmethod
    def _failure(job_id: str, model_type: str, model_name: str, error: str, ts: Optional[int] = None) -> dict:
        return {
            "job_id": job_id,
            "model_type": model_type,
            "model_name": model_name,
            "error": error[:MAX_ERROR_LENGTH],
            "ts": ts if ts is not None else int(time.time() * 1000),
        }

    def dead_workers(self) -> List[Tuple[str, object, str]]:
        """(job_id, worker, model_type) for every worker that started OK
        but whose thread has since died — the watchdog's restart list."""
        dead: List[Tuple[str, object, str]] = []
        with self._lock:
            for job_id, entries in self._entries.items():
                if not self._worker_manager.is_running(job_id):
                    continue
                for entry in entries:
                    if entry.started and entry.start_error is None and not entry.worker.is_alive():
                        if entry.died_at is None:
                            entry.died_at = int(time.time() * 1000)
                        dead.append((job_id, entry.worker, entry.model_type))
        return dead

    def snapshot(self) -> Tuple[Dict[str, List[str]], List[dict]]:
        """(running_jobs, failed_assignments). A model counts as running
        only while its worker thread is actually alive."""
        running: Dict[str, List[str]] = {}
        failed: List[dict] = []
        self.dead_workers()  # stamps died_at so the timestamps below are stable
        with self._lock:
            for entries in self._failed.values():
                failed.extend(dict(e) for e in entries)
            for job_id, entries in self._entries.items():
                job_running = self._worker_manager.is_running(job_id)
                for entry in entries:
                    if entry.start_error is not None:
                        failed.append(self._failure(job_id, entry.model_type, entry.model_name, "start failed: " + entry.start_error))
                    elif not job_running or not entry.started:
                        continue
                    elif entry.worker.is_alive():
                        running.setdefault(job_id, []).append(entry.model_type)
                    else:
                        failed.append(self._failure(job_id, entry.model_type, entry.model_name, "worker stopped unexpectedly", entry.died_at))
        return running, failed

    def camera_status(self, job_id: str) -> str:
        """'online' / 'offline' — read from the live RTSP connection."""
        return self._worker_manager.camera_status(job_id)
