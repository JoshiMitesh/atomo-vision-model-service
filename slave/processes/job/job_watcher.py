"""Poll jobs.yaml every N seconds and diff against the previous snapshot.

Pure diffing logic lives here — job_watcher.py has no idea how to start
or stop a worker, it only classifies ADDED / REMOVED / CHANGED /
UNCHANGED and hands the result to a callback (job_manager.py).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List

from slave.logging.logger import get_logger
from slave.models.job import JobConfig
from slave.storage.yaml import job_store

log = get_logger(__name__)


class JobDiffType(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class JobDiff:
    diff_type: JobDiffType
    job_id: str
    job: JobConfig | None  # None only for REMOVED


DiffCallback = Callable[[List[JobDiff]], None]


def diff_jobs(previous: Dict[str, JobConfig], current: List[JobConfig]) -> List[JobDiff]:
    """Compare the previous snapshot against a freshly loaded job list."""
    diffs: List[JobDiff] = []
    current_by_id = {job.job_id: job for job in current}

    for job_id, prev_job in previous.items():
        if job_id not in current_by_id:
            diffs.append(JobDiff(JobDiffType.REMOVED, job_id, None))

    for job_id, job in current_by_id.items():
        prev_job = previous.get(job_id)
        if prev_job is None:
            diffs.append(JobDiff(JobDiffType.ADDED, job_id, job))
        elif prev_job.config_changed(job):
            diffs.append(JobDiff(JobDiffType.CHANGED, job_id, job))
        else:
            diffs.append(JobDiff(JobDiffType.UNCHANGED, job_id, job))

    return diffs


class JobWatcher:
    """Background thread that polls jobs.yaml on a fixed interval."""

    def __init__(self, jobs_yaml_path: str, poll_interval_seconds: int, on_diff: DiffCallback) -> None:
        self._jobs_yaml_path = jobs_yaml_path
        self._poll_interval_seconds = poll_interval_seconds
        self._on_diff = on_diff
        self._snapshot: Dict[str, JobConfig] = {}
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="job-watcher", daemon=True)

    def start(self) -> None:
        log.info("job watcher starting (poll interval=%ss)", self._poll_interval_seconds)
        self._poll_once()  # apply initial state immediately on startup
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=self._poll_interval_seconds + 5)
        log.info("job watcher stopped")

    def _run(self) -> None:
        while not self._stop_event.wait(self._poll_interval_seconds):
            self._poll_once()

    def _poll_once(self) -> None:
        try:
            current = job_store.load_jobs(self._jobs_yaml_path)
        except Exception:
            log.exception("failed to read jobs.yaml, skipping this poll cycle")
            return

        diffs = diff_jobs(self._snapshot, current)
        self._snapshot = {job.job_id: job for job in current}

        actionable = [d for d in diffs if d.diff_type is not JobDiffType.UNCHANGED]
        if actionable:
            log.info(
                "jobs.yaml diff: %d added, %d removed, %d changed",
                sum(1 for d in diffs if d.diff_type is JobDiffType.ADDED),
                sum(1 for d in diffs if d.diff_type is JobDiffType.REMOVED),
                sum(1 for d in diffs if d.diff_type is JobDiffType.CHANGED),
            )
            self._on_diff(diffs)
