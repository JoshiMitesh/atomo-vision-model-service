"""config/jobs.yaml persistence.

jobs.yaml is *runtime state* (CLAUDE.md §3), not static config. This
module is the only code allowed to write to it — the NATS subscriber
and stop handler call into it, job_watcher.py only ever reads it.

All writes are serialized with a lock and go through an atomic
write-then-replace so a crash mid-write can never leave jobs.yaml
truncated or corrupted.
"""

from __future__ import annotations

import threading
from typing import List

import yaml

from slave.models.job import JobConfig
from slave.utils.file_utils import atomic_write_text

_lock = threading.Lock()


def _read_raw(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError:
        return {"jobs": []}

    if data is None:
        return {"jobs": []}
    if isinstance(data, list):
        return {"jobs": data}
    if isinstance(data, dict):
        return data
    return {"jobs": []}


def load_jobs(path: str) -> List[JobConfig]:
    """Read all jobs currently in jobs.yaml. Never raises on a missing
    or empty file — an absent jobs.yaml simply means zero active jobs.
    """
    with _lock:
        raw = _read_raw(path)
    entries = raw.get("jobs") or []
    return [JobConfig.from_dict(entry) for entry in entries]


def save_jobs(path: str, jobs: List[JobConfig]) -> None:
    with _lock:
        payload = {"jobs": [job.to_dict() for job in jobs]}
        atomic_write_text(path, yaml.safe_dump(payload, sort_keys=False))


def upsert_job(path: str, job: JobConfig) -> None:
    """Add job, or replace the existing entry with the same job_id."""
    with _lock:
        raw = _read_raw(path)
        entries = raw.get("jobs") or []
        entries = [e for e in entries if str(e.get("job_id")) != job.job_id]
        entries.append(job.to_dict())
        atomic_write_text(path, yaml.safe_dump({"jobs": entries}, sort_keys=False))


def remove_job(path: str, job_id: str) -> bool:
    """Remove job_id from jobs.yaml. Returns True if it was present."""
    with _lock:
        raw = _read_raw(path)
        entries = raw.get("jobs") or []
        remaining = [e for e in entries if str(e.get("job_id")) != job_id]
        removed = len(remaining) != len(entries)
        if removed:
            atomic_write_text(path, yaml.safe_dump({"jobs": remaining}, sort_keys=False))
        return removed
