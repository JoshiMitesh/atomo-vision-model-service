"""Turns job_watcher.py diffs into worker start/stop/restart actions.

Each diff is handled independently and defensively — a failure
building or stopping one job's worker must never affect any other job
(CLAUDE.md §20, specs.md §20).
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Tuple

from slave.logging.logger import get_logger
from slave.processes.device.runtime_status import RuntimeStatus
from slave.models.job import JobConfig
from slave.processes.job.job_watcher import JobDiff, JobDiffType
from slave.processes.model.model_manager import ModelManager
from slave.processes.worker.worker_manager import WorkerManager

log = get_logger(__name__)


class JobManager:
    def __init__(
        self,
        worker_manager: WorkerManager,
        model_manager: ModelManager,
        runtime_status: Optional[RuntimeStatus] = None,
        watchdog_interval_seconds: float = 10.0,
        max_restarts_per_worker: int = 3,
        restart_backoff_seconds: float = 30.0,
    ) -> None:
        self._worker_manager = worker_manager
        self._model_manager = model_manager
        self._runtime_status = runtime_status
        self._jobs: Dict[str, JobConfig] = {}
        self._watchdog_interval = watchdog_interval_seconds
        self._max_restarts = max_restarts_per_worker
        self._restart_backoff = restart_backoff_seconds
        self._restarts: Dict[Tuple[str, str], Tuple[int, float]] = {}  # (job_id, model_type) -> (count, last_try)
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: Optional[threading.Thread] = None

    def handle_diffs(self, diffs: List[JobDiff]) -> None:
        for diff in diffs:
            try:
                if diff.diff_type is JobDiffType.ADDED:
                    self._handle_added(diff)
                elif diff.diff_type is JobDiffType.REMOVED:
                    self._handle_removed(diff)
                elif diff.diff_type is JobDiffType.CHANGED:
                    self._handle_changed(diff)
            except Exception:
                log.exception("failed to apply %s for job_id=%s", diff.diff_type.value, diff.job_id)

    def _handle_added(self, diff: JobDiff) -> None:
        assert diff.job is not None
        workers = self._model_manager.build_workers(diff.job)
        self._jobs[diff.job_id] = diff.job
        self._clear_restarts(diff.job_id)
        failures = self._worker_manager.start_worker(diff.job_id, workers)
        if self._runtime_status is not None:
            self._runtime_status.job_started(diff.job_id, failures)

    def _handle_removed(self, diff: JobDiff) -> None:
        self._worker_manager.stop_worker(diff.job_id)
        self._jobs.pop(diff.job_id, None)
        self._clear_restarts(diff.job_id)
        if self._runtime_status is not None:
            self._runtime_status.job_removed(diff.job_id)

    def _handle_changed(self, diff: JobDiff) -> None:
        assert diff.job is not None
        workers = self._model_manager.build_workers(diff.job)
        self._jobs[diff.job_id] = diff.job
        self._clear_restarts(diff.job_id)
        failures = self._worker_manager.restart_worker(diff.job_id, workers)
        if self._runtime_status is not None:
            self._runtime_status.job_started(diff.job_id, failures)

    # -- watchdog: restart a model worker whose thread died ------------------

    def start_watchdog(self) -> None:
        if self._runtime_status is None or self._watchdog_thread is not None:
            return
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, name="worker-watchdog", daemon=True)
        self._watchdog_thread.start()

    def stop_watchdog(self) -> None:
        self._watchdog_stop.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=5)
            self._watchdog_thread = None

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.wait(self._watchdog_interval):
            try:
                self.check_workers()
            except Exception:
                log.exception("worker watchdog tick failed")

    def check_workers(self) -> None:
        """Restart each dead worker on its own (siblings keep running).
        At most `max_restarts_per_worker` attempts per model, spaced by
        `restart_backoff_seconds`, so a model that crashes immediately
        every time can't loop forever — it stays reported in
        failed_assignments instead.
        """
        assert self._runtime_status is not None
        for job_id, dead_worker, model_type in self._runtime_status.dead_workers():
            job = self._jobs.get(job_id)
            assignment = job.assignment_for(model_type) if job else None
            if job is None or assignment is None:
                continue
            count, last = self._restarts.get((job_id, model_type), (0, 0.0))
            now = time.monotonic()
            if count >= self._max_restarts or now - last < self._restart_backoff:
                continue
            self._restarts[(job_id, model_type)] = (count + 1, now)
            log.warning("job_id=%s: %s worker died, restarting (attempt %d/%d)", job_id, model_type, count + 1, self._max_restarts)
            new_worker = self._model_manager.build_one(job, assignment)
            if new_worker is None:
                continue
            exc = self._worker_manager.replace_worker(job_id, dead_worker, new_worker)
            self._runtime_status.replace_worker(
                job_id, dead_worker, new_worker, None if exc is None else f"{type(exc).__name__}: {exc}"
            )

    def _clear_restarts(self, job_id: str) -> None:
        for key in [k for k in self._restarts if k[0] == job_id]:
            del self._restarts[key]
