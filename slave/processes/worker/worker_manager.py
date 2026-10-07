"""Tracks the running worker(s) per job_id.

Every Person/Face/Safety/Fire worker (pt/nb/so) exposes the same small
lifecycle contract — start(), stop(), is_alive() — without sharing a
base class (AGENTS.md §7: "the only common contract should be the
final detection result"). WorkerManager only relies on that duck-typed
contract.

A job_id can now own more than one worker at once — one per
model_assignments[] entry (person + face + safety + fire all running
concurrently on the same job/camera) — so each job_id maps to a LIST
of workers, all started/stopped/restarted together as one unit. A job
change (CHANGED) still tears down every worker for that job_id and
rebuilds the full list from the new config, same as before; there's no
partial per-assignment diffing.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Protocol

from slave.logging.logger import get_logger

log = get_logger(__name__)


class Worker(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def is_alive(self) -> bool: ...


class WorkerManager:
    def __init__(self) -> None:
        self._workers: Dict[str, List[Worker]] = {}
        self._lock = threading.Lock()

    def start_worker(self, job_id: str, workers: List[Worker]) -> None:
        with self._lock:
            if job_id in self._workers:
                log.warning("worker(s) already running for job_id=%s, refusing duplicate start", job_id)
                return
            self._workers[job_id] = list(workers)
        for worker in workers:
            worker.start()
        log.info("%d worker(s) started for job_id=%s", len(workers), job_id)

    def stop_worker(self, job_id: str) -> None:
        with self._lock:
            workers = self._workers.pop(job_id, None)
        if not workers:
            log.warning("stop requested for job_id=%s but no worker is running", job_id)
            return
        for worker in workers:
            try:
                worker.stop()
            except Exception:
                log.exception("error stopping a worker for job_id=%s", job_id)
        log.info("%d worker(s) stopped for job_id=%s", len(workers), job_id)

    def restart_worker(self, job_id: str, workers: List[Worker]) -> None:
        """Stop all old workers for job_id and, if it was previously
        running, replace with the new configuration (used for CHANGED
        jobs).
        """
        self.stop_worker(job_id)
        self.start_worker(job_id, workers)

    def is_running(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._workers

    def camera_status(self, job_id: str) -> str:
        """'online' if any of the job's workers has a live RTSP
        connection, else 'offline' (no worker, or stream down/
        reconnecting). Reads each worker's FrameReader (every worker
        owns exactly one, as `_frame_reader`).
        """
        with self._lock:
            workers = list(self._workers.get(job_id, []))
        for worker in workers:
            reader = getattr(worker, "_frame_reader", None)
            if reader is not None and reader.get_status() == "online":
                return "online"
        return "offline"

    def stop_all(self) -> None:
        with self._lock:
            job_ids = list(self._workers.keys())
        for job_id in job_ids:
            self.stop_worker(job_id)
        log.info("all workers stopped")
