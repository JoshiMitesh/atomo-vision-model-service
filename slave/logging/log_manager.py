"""Process-wide logging configuration — the only module that touches
handlers/formatters (CLAUDE.md §19).

Every log record is written twice:

    logging_output/<process-area>.log     — e.g. subscriber.log, job.log, worker.log
    logging_output/jobs/<job_id>.log      — everything tagged with that job_id

both under ONE directory, always anchored inside this `slave/` package
(never dependent on the process's current working directory, and never
outside `slave/`).

`main.py` calls ``LogManager.init(...)`` exactly once, early, with the
real settings from config/slave.yaml. Every other module just calls
``logger.get_logger(__name__)`` — that never needs LogManager to be
initialized first: child loggers propagate up to the "slave" root
logger and pick up its handlers whenever they're attached, regardless
of import order.
"""

from __future__ import annotations

import logging as stdlib_logging
import os
import re
import threading
from typing import Dict, Optional

from .rotating_file_handler import build_rotating_file_handler

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | job_id=%(job_id)s | %(message)s"

# slave/logging/log_manager.py -> slave/
_SLAVE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LOG_DIR = os.path.join(_SLAVE_ROOT, "logging_output")

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def _area_for(logger_name: str) -> str:
    """Map 'slave.<...>' to a process-area file name, mirroring the
    processes/ layout, e.g.:
        slave.processes.subscriber.subscriber -> subscriber
        slave.processes.job.job_watcher        -> job
        slave.processes.worker.worker_manager  -> worker
        slave.storage.yaml.job_store           -> storage
        slave.main                             -> main
    """
    parts = logger_name.split(".")
    if len(parts) >= 3 and parts[1] == "processes":
        return parts[2]
    if len(parts) >= 2:
        return parts[1]
    return "main"


class _JobIdDefaultFilter(stdlib_logging.Filter):
    """Guarantee every record has a job_id, even outside job context."""

    def filter(self, record: stdlib_logging.LogRecord) -> bool:
        if not hasattr(record, "job_id"):
            record.job_id = "-"
        return True


class _RoutingHandler(stdlib_logging.Handler):
    """Fans each record out to its process-area file and, if tagged
    with a job_id, to that job's own file too. Per-file handlers are
    created lazily and each rotates independently at ``max_bytes``.
    """

    def __init__(self, log_dir: str, max_bytes: int, backup_count: int) -> None:
        super().__init__()
        self._log_dir = log_dir
        self._jobs_dir = os.path.join(log_dir, "jobs")
        os.makedirs(self._jobs_dir, exist_ok=True)
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        self._formatter = stdlib_logging.Formatter(_LOG_FORMAT)
        self._area_handlers: Dict[str, stdlib_logging.Handler] = {}
        self._job_handlers: Dict[str, stdlib_logging.Handler] = {}
        self._lock = threading.Lock()

    def _get_or_create(self, cache: Dict[str, stdlib_logging.Handler], key: str, path: str) -> stdlib_logging.Handler:
        handler = cache.get(key)
        if handler is not None:
            return handler
        with self._lock:
            handler = cache.get(key)
            if handler is None:
                handler = build_rotating_file_handler(path, self._max_bytes, self._backup_count)
                handler.setFormatter(self._formatter)
                cache[key] = handler
            return handler

    def emit(self, record: stdlib_logging.LogRecord) -> None:
        try:
            area = _area_for(record.name)
            self._get_or_create(
                self._area_handlers, area, os.path.join(self._log_dir, f"{area}.log")
            ).emit(record)

            job_id = getattr(record, "job_id", "-")
            if job_id and job_id != "-":
                safe_id = _UNSAFE_CHARS.sub("_", job_id)
                self._get_or_create(
                    self._job_handlers, safe_id, os.path.join(self._jobs_dir, f"{safe_id}.log")
                ).emit(record)
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        with self._lock:
            for handler in list(self._area_handlers.values()) + list(self._job_handlers.values()):
                handler.close()
        super().close()


class LogManager:
    _lock = threading.Lock()
    _configured = False

    def __init__(self, log_dir: str, max_bytes: int, backup_count: int, level: str) -> None:
        os.makedirs(log_dir, exist_ok=True)
        root = stdlib_logging.getLogger("slave")
        root.setLevel(getattr(stdlib_logging, level.upper(), stdlib_logging.INFO))
        root.propagate = False

        if not root.handlers:
            job_filter = _JobIdDefaultFilter()

            routing_handler = _RoutingHandler(log_dir, max_bytes, backup_count)
            routing_handler.addFilter(job_filter)
            root.addHandler(routing_handler)

            console_handler = stdlib_logging.StreamHandler()
            console_handler.setFormatter(stdlib_logging.Formatter(_LOG_FORMAT))
            console_handler.addFilter(job_filter)
            root.addHandler(console_handler)

    @classmethod
    def init(
        cls,
        log_dir: Optional[str] = None,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        level: str = "INFO",
    ) -> None:
        """Configure the "slave" logger tree. Safe to call more than
        once — only the first call has any effect.
        """
        with cls._lock:
            if cls._configured:
                return
            cls(log_dir or DEFAULT_LOG_DIR, max_bytes, backup_count, level)
            cls._configured = True
