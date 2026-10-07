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
from typing import Dict, List, Optional, Protocol, Tuple

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

    def start_worker(self, job_id: str, workers: List[Worker]) -> List[Tuple[Worker, Exception]]:
        """Start every worker for job_id independently: one worker whose
        start() raises is dropped and reported, the rest still start.
        Returns the (worker, exception) pairs that failed to start.
        """
        with self._lock:
            if job_id in self._workers:
                log.warning("worker(s) already running for job_id=%s, refusing duplicate start", job_id)
                return []
            self._workers[job_id] = list(workers)
        failures: List[Tuple[Worker, Exception]] = []
        for worker in workers:
            try:
                worker.start()
            except Exception as exc:
                log.exception("job_id=%s: a worker failed to start, continuing with the others", job_id)
                failures.append((worker, exc))
                try:
                    worker.stop()  # release anything it half-acquired
                except Exception:
                    pass
        if failures:
            failed_ids = {id(w) for w, _ in failures}
            with self._lock:
                remaining = [w for w in self._workers.get(job_id, []) if id(w) not in failed_ids]
                if remaining:
                    self._workers[job_id] = remaining
                else:
                    self._workers.pop(job_id, None)
        log.info("%d/%d worker(s) started for job_id=%s", len(workers) - len(failures), len(workers), job_id)
        return failures

    def replace_worker(self, job_id: str, old: Worker, new: Worker) -> Optional[Exception]:
        """Swap one dead worker for a freshly built one and start it.
        Returns None on success, or the exception if the new worker
        failed to start (it is then dropped; siblings are untouched).
        """
        with self._lock:
            current = self._workers.get(job_id)
            if current is None or old not in current:
                return RuntimeError("job or worker no longer present")
            current[current.index(old)] = new
        try:
            old.stop()
        except Exception:
            log.exception("job_id=%s: error stopping dead worker before restart", job_id)
        try:
            new.start()
        except Exception as exc:
            log.exception("job_id=%s: restarted worker failed to start", job_id)
            try:
                new.stop()
            except Exception:
                pass
            with self._lock:
                current = self._workers.get(job_id)
                if current is not None and new in current:
                    current.remove(new)
            return exc
        return None

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

    def restart_worker(self, job_id: str, workers: List[Worker]) -> List[Tuple[Worker, Exception]]:
        """Stop all old workers for job_id and, if it was previously
        running, replace with the new configuration (used for CHANGED
        jobs).
        """
        self.stop_worker(job_id)
        return self.start_worker(job_id, workers)

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
