"""Public logging API used by every process/worker in the Slave.

Usage:
    from slave.logging.logger import get_logger
    log = get_logger(__name__, job_id="job_001")
    log.info("worker started")

Safe to call at import time, before main.py runs LogManager.init() —
this just returns a child logger under the "slave" namespace; it picks
up real handlers (and starts actually writing anywhere) once
LogManager.init() attaches them to the "slave" root logger, regardless
of when that happens relative to this call.
"""

from __future__ import annotations

import logging as stdlib_logging
from typing import Optional


def get_logger(name: str, job_id: Optional[str] = None) -> stdlib_logging.LoggerAdapter:
    # Every internal caller passes __name__, which is already fully
    # qualified as "slave.processes...." — don't double-prefix it.
    # Only bare/external names (e.g. "worker") get "slave." added.
    full_name = name if name == "slave" or name.startswith("slave.") else f"slave.{name}"
    base = stdlib_logging.getLogger(full_name)
    return stdlib_logging.LoggerAdapter(base, {"job_id": job_id or "-"})
