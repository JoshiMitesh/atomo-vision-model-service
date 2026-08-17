"""Deterministic model_type + model_extension -> implementation path.

person + .pt  -> processes/worker/models/person/pt/
person + .nb  -> processes/worker/models/person/nb/
person + .so  -> processes/worker/models/person/so/

Anything else (face, fire, fight, unknown_model, unsupported
extensions) is rejected here, before any worker is built
(CLAUDE.md §7).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from slave.models.job import JobConfig
from slave.processes.model.model_registry import (
    SUPPORTED_EXTENSIONS,
    SUPPORTED_MODEL_TYPES,
    ModelNotSupportedError,
)

_MODEL_IMPL_DIR = {
    (".pt",): "pt",
    (".nb",): "nb",
    (".so",): "so",
}


@dataclass(frozen=True)
class ResolvedModel:
    model_type: str
    model_extension: str
    impl_subdir: str          # e.g. "pt" | "nb" | "so"
    model_config_path: str    # e.g. config/models/person.yaml


def resolve(job: JobConfig, config_root: str) -> ResolvedModel:
    if job.model_type not in SUPPORTED_MODEL_TYPES:
        raise ModelNotSupportedError(
            f"job {job.job_id}: model_type='{job.model_type}' is not implemented "
            f"(supported: {SUPPORTED_MODEL_TYPES})"
        )
    if job.model_extension not in SUPPORTED_EXTENSIONS:
        raise ModelNotSupportedError(
            f"job {job.job_id}: model_extension='{job.model_extension}' is not supported "
            f"(supported: {SUPPORTED_EXTENSIONS})"
        )

    impl_subdir = job.model_extension.lstrip(".")
    model_config_path = os.path.join(config_root, "models", f"{job.model_type}.yaml")

    return ResolvedModel(
        model_type=job.model_type,
        model_extension=job.model_extension,
        impl_subdir=impl_subdir,
        model_config_path=model_config_path,
    )
