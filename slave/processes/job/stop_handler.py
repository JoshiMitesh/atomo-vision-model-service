"""Apply a validated stop request to jobs.yaml.

Sits between stop_message.py (parsing) and job_store.py (persistence)
so failures are contained here and logged instead of propagating up
into the NATS callback or corrupting jobs.yaml (AGENTS.md §18).
"""

from __future__ import annotations

from slave.logging.logger import get_logger
from slave.storage.yaml import job_store

log = get_logger(__name__)


def handle_stop(jobs_yaml_path: str, job_id: str) -> bool:
    """Remove job_id from jobs.yaml. Returns True if a job was removed.

    Never raises — a storage failure here must not take down the
    subscriber or the rest of the Slave process.
    """
    try:
        removed = job_store.remove_job(jobs_yaml_path, job_id)
        if removed:
            log.info("stop request applied, job removed from jobs.yaml (job_id=%s)", job_id)
        else:
            log.warning("stop request for unknown job_id (no-op) (job_id=%s)", job_id)
        return removed
    except Exception:
        log.exception("failed to apply stop request (job_id=%s)", job_id)
        return False
