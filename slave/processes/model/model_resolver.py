"""Deterministic model_type + model_extension -> implementation path.

person + .pt  -> processes/worker/models/person/pt/
person + .nb  -> processes/worker/models/person/nb/
person + .so  -> processes/worker/models/person/so/

Same pattern for face/safety/fire, each under its own
processes/worker/models/<type>/{pt,nb,so}/ (see model_registry.py's
SUPPORTED_MODEL_TYPES for the current list). Anything else (fight,
unknown_model, unsupported extensions) is rejected here, before any
worker is built (CLAUDE.md §7).

Resolves one ModelAssignment at a time now, not a whole job — a job
can carry several assignments (person + face + safety + fire all on
one camera) and model_manager.py calls this once per assignment to
build one worker each.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from slave.models.job import ModelAssignment
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


def resolve(assignment: ModelAssignment, job_id: str, config_root: str) -> ResolvedModel:
    if assignment.model_type not in SUPPORTED_MODEL_TYPES:
        raise ModelNotSupportedError(
            f"job {job_id}: model_type='{assignment.model_type}' is not implemented "
            f"(supported: {SUPPORTED_MODEL_TYPES})"
        )
    if assignment.model_extension not in SUPPORTED_EXTENSIONS:
        raise ModelNotSupportedError(
            f"job {job_id}: model_extension='{assignment.model_extension}' is not supported "
            f"(supported: {SUPPORTED_EXTENSIONS})"
        )

    impl_subdir = assignment.model_extension.lstrip(".")
    model_config_path = os.path.join(config_root, "models", f"{assignment.model_type}.yaml")

    return ResolvedModel(
        model_type=assignment.model_type,
        model_extension=assignment.model_extension,
        impl_subdir=impl_subdir,
        model_config_path=model_config_path,
    )
