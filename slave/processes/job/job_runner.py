"""Composition root for the job subsystem.

Wires JobWatcher (polling/diffing) to JobManager (start/stop/restart
logic) and exposes a single start()/stop() pair for main.py.
"""

from __future__ import annotations

from slave.logging.logger import get_logger
from slave.processes.job.job_manager import JobManager
from slave.processes.job.job_watcher import JobWatcher

log = get_logger(__name__)


class JobRunner:
    def __init__(self, jobs_yaml_path: str, poll_interval_seconds: int, job_manager: JobManager) -> None:
        self._job_manager = job_manager
        self._watcher = JobWatcher(
            jobs_yaml_path=jobs_yaml_path,
            poll_interval_seconds=poll_interval_seconds,
            on_diff=job_manager.handle_diffs,
        )

    def start(self) -> None:
        self._watcher.start()
        log.info("job runner started")

    def stop(self) -> None:
        self._watcher.stop()
        log.info("job runner stopped")
