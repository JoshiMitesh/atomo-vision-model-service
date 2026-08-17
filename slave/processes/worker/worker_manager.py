"""Tracks the one running worker per job_id.

Every Person worker (pt/nb/so) exposes the same small lifecycle
contract — start(), stop(), is_alive() — without sharing a base class
(AGENTS.md §7: "the only common contract should be the final
detection result"). WorkerManager only relies on that duck-typed
contract.
"""

from __future__ import annotations

import threading
from typing import Dict, Protocol

from slave.logging.logger import get_logger

log = get_logger(__name__)


class Worker(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def is_alive(self) -> bool: ...


class WorkerManager:
    def __init__(self) -> None:
        self._workers: Dict[str, Worker] = {}
        self._lock = threading.Lock()

    def start_worker(self, job_id: str, worker: Worker) -> None:
        with self._lock:
            if job_id in self._workers:
                log.warning("worker already running for job_id=%s, refusing duplicate start", job_id)
                return
            self._workers[job_id] = worker
        worker.start()
        log.info("worker started for job_id=%s", job_id)

    def stop_worker(self, job_id: str) -> None:
        with self._lock:
            worker = self._workers.pop(job_id, None)
        if worker is None:
            log.warning("stop requested for job_id=%s but no worker is running", job_id)
            return
        worker.stop()
        log.info("worker stopped for job_id=%s", job_id)

    def restart_worker(self, job_id: str, worker: Worker) -> None:
        """Stop old worker for job_id and if it was previously running,
        replace with the new configuration (used for CHANGED jobs).
        """
        self.stop_worker(job_id)
        self.start_worker(job_id, worker)

    def is_running(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._workers

    def stop_all(self) -> None:
        with self._lock:
            job_ids = list(self._workers.keys())
        for job_id in job_ids:
            self.stop_worker(job_id)
        log.info("all workers stopped")
