"""Top-level facade for turning a JobConfig into its running workers.

    JobConfig (model_assignments[])
        ↓  (once per assignment)
    model_resolver  (validate + compute paths)
        ↓
    model_registry  (look up the worker class)
        ↓
    worker_factory  (construct the worker instance)

A job can carry several assignments (person + face + safety + fire
all on one camera) — build_workers() builds one worker per assignment
so all of them run concurrently against the job's camera_stream. One
bad assignment (unsupported model_type, missing model file, etc.)
must not take the rest of the job down with it (CLAUDE.md §20's
per-job isolation, applied at assignment granularity here) — it's
logged and skipped instead of raising.

job_manager.py is the only caller of this module.
"""

from __future__ import annotations

from typing import List, Optional

import yaml

from slave.logging.logger import get_logger
from slave.models.job import JobConfig
from slave.processes.device.runtime_status import RuntimeStatus
from slave.processes.model.model_registry import ModelRegistry
from slave.processes.model.model_resolver import ResolvedModel, resolve
from slave.processes.worker import worker_factory
from slave.processes.worker.worker_factory import WorkerServices

log = get_logger(__name__)


class ModelManager:
    def __init__(
        self,
        registry: ModelRegistry,
        config_root: str,
        services: WorkerServices,
        runtime_status: Optional[RuntimeStatus] = None,
    ) -> None:
        self._registry = registry
        self._config_root = config_root
        self._services = services
        self._runtime_status = runtime_status

    def build_workers(self, job: JobConfig) -> List[object]:
        workers: List[object] = []
        built_types: List[str] = []
        if self._runtime_status is not None:
            self._runtime_status.reset_failures(job.job_id)
        for assignment in job.model_assignments:
            try:
                resolved: ResolvedModel = resolve(assignment, job.job_id, self._config_root)
                worker_cls = self._registry.get(resolved.model_type, resolved.model_extension)
                model_config = self._load_model_config(resolved.model_config_path)
                worker = worker_factory.create_worker(
                    job, assignment, resolved, worker_cls, model_config, self._services
                )
            except Exception as exc:
                log.exception(
                    "job %s: failed to build worker for model_type=%s, skipping this assignment",
                    job.job_id,
                    assignment.model_type,
                )
                if self._runtime_status is not None:
                    self._runtime_status.record_failure(
                        job.job_id, assignment.model_type, assignment.model_name, f"{type(exc).__name__}: {exc}"
                    )
                continue
            workers.append(worker)
            built_types.append(assignment.model_type)
        if self._runtime_status is not None:
            self._runtime_status.job_built(job.job_id, built_types)
        return workers

    @staticmethod
    def _load_model_config(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
