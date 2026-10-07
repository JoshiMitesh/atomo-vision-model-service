"""Turns job_watcher.py diffs into worker start/stop/restart actions.

Each diff is handled independently and defensively — a failure
building or stopping one job's worker must never affect any other job
(CLAUDE.md §20, specs.md §20).
"""

from __future__ import annotations

from typing import List, Optional

from slave.logging.logger import get_logger
from slave.processes.device.runtime_status import RuntimeStatus
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
    ) -> None:
        self._worker_manager = worker_manager
        self._model_manager = model_manager
        self._runtime_status = runtime_status

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
        self._worker_manager.start_worker(diff.job_id, workers)

    def _handle_removed(self, diff: JobDiff) -> None:
        self._worker_manager.stop_worker(diff.job_id)
        if self._runtime_status is not None:
            self._runtime_status.job_removed(diff.job_id)

    def _handle_changed(self, diff: JobDiff) -> None:
        assert diff.job is not None
        workers = self._model_manager.build_workers(diff.job)
        self._worker_manager.restart_worker(diff.job_id, workers)
