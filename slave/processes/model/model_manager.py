"""Top-level facade for turning a JobConfig into a running worker.

    JobConfig
        ↓
    model_resolver  (validate + compute paths)
        ↓
    model_registry  (look up the worker class)
        ↓
    worker_factory  (construct the worker instance)

job_manager.py is the only caller of this module.
"""

from __future__ import annotations

import yaml

from slave.models.job import JobConfig
from slave.processes.model.model_registry import ModelRegistry
from slave.processes.model.model_resolver import ResolvedModel, resolve
from slave.processes.worker import worker_factory
from slave.processes.worker.worker_factory import WorkerServices


class ModelManager:
    def __init__(self, registry: ModelRegistry, config_root: str, services: WorkerServices) -> None:
        self._registry = registry
        self._config_root = config_root
        self._services = services

    def build_worker(self, job: JobConfig):
        resolved: ResolvedModel = resolve(job, self._config_root)
        worker_cls = self._registry.get(resolved.model_type, resolved.model_extension)
        model_config = self._load_model_config(resolved.model_config_path)
        return worker_factory.create_worker(job, resolved, worker_cls, model_config, self._services)

    @staticmethod
    def _load_model_config(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
