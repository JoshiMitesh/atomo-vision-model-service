"""JobConfig — the in-memory representation of one entry in jobs.yaml.

Mirrors the job-start message Master sends on slave.jobs.start (see
processes/subscriber/job_message.py). There is no single top-level
model_name/model_type anymore — a job's camera can run more than one
model at once (e.g. person + face), so each model lives in its own
`model_assignments[]` entry: {model_type, model_name,
camera_model_assign_id}.

Which assignment actually drives the running worker (model_resolver.py
/ model_registry.py — one worker class per job) is `primary_assignment`
below: "person" wins whenever present (face rides along on the same
frame via face_controller.py, same as before), otherwise the job's
only assignment (e.g. a standalone Face job) is primary.

master_ip/stream_port/stream_name are also gone — the Slave is already
NATS-connected to Master. Master sends the connectable RTSP URL
directly as `camera_stream` (any host/port/path); `camera_id` is an
identity/label only, carried through to detections/telemetry/MinIO
paths, and plays no part in connecting to the stream. `camera_name` is
a second, human-readable identity field Master sends alongside
`camera_id` — also carried through untouched, published in every
DetectionResult's `source.camera`.

Each model_assignments[] entry also carries `model_id` — Master's
catalog id for that model (distinct from `camera_model_assign_id`,
which identifies *this camera's assignment* of the model, not the
model itself). Published in every ClassDetection right after
camera_model_assign_id — see models/detection.py.

`zone_name` is a second, human-readable identity field Master sends
alongside `zone_id` — carried through untouched, published in every
DetectionResult's `source` right after `zone_id`.

`branch_id`/`branch_name` are also Master-supplied, per-job (a device
can run jobs for more than one branch) — NOT device-level config, so
they don't live in slave.yaml/DeviceConfig. Published in every
DetectionResult's `source`, right before `device_id`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple


def _optional_float(value) -> Optional[float]:
    return None if value is None else float(value)


@dataclass(frozen=True)
class ModelAssignment:
    """One model Master has assigned to this job's camera."""

    model_type: str
    model_name: str
    camera_model_assign_id: str
    model_id: str

    @property
    def model_extension(self) -> str:
        return os.path.splitext(self.model_name)[1]

    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "camera_model_assign_id": self.camera_model_assign_id,
            "model_id": self.model_id,
        }

    @staticmethod
    def from_dict(raw: dict) -> "ModelAssignment":
        return ModelAssignment(
            model_type=str(raw["model_type"]),
            model_name=str(raw["model_name"]),
            camera_model_assign_id=str(raw["camera_model_assign_id"]),
            model_id=str(raw["model_id"]),
        )


@dataclass(frozen=True)
class JobConfig:
    job_id: str
    branch_id: str
    branch_name: str
    camera_id: str
    camera_name: str
    camera_stream: str
    zone_id: str
    zone_name: str
    model_assignments: Tuple[ModelAssignment, ...]
    # Optional camera location, Master-supplied alongside the other
    # camera_* identity fields. Only surfaced in the device health
    # message's cameras[] entry (null when the job carries none).
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    @property
    def primary_assignment(self) -> ModelAssignment:
        """The assignment that determines which worker class runs this
        job. "person" wins whenever present (face rides along on the
        same frame via face_controller.py); otherwise the job's own
        first (and normally only) assignment is primary — e.g. a
        standalone Face job.
        """
        for assignment in self.model_assignments:
            if assignment.model_type == "person":
                return assignment
        if not self.model_assignments:
            raise ValueError(f"job {self.job_id} has no model_assignments")
        return self.model_assignments[0]

    @property
    def model_type(self) -> str:
        return self.primary_assignment.model_type

    @property
    def model_name(self) -> str:
        return self.primary_assignment.model_name

    @property
    def model_extension(self) -> str:
        return self.primary_assignment.model_extension

    def assignment_for(self, model_type: str) -> Optional[ModelAssignment]:
        for assignment in self.model_assignments:
            if assignment.model_type == model_type:
                return assignment
        return None

    def camera_model_assign_id_for(self, model_type: str) -> Optional[str]:
        assignment = self.assignment_for(model_type)
        return assignment.camera_model_assign_id if assignment else None

    def model_id_for(self, model_type: str) -> Optional[str]:
        assignment = self.assignment_for(model_type)
        return assignment.model_id if assignment else None

    def model_name_for(self, model_type: str) -> Optional[str]:
        assignment = self.assignment_for(model_type)
        return assignment.model_name if assignment else None

    def to_dict(self) -> dict:
        result = {
            "job_id": self.job_id,
            "branch_id": self.branch_id,
            "branch_name": self.branch_name,
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "camera_stream": self.camera_stream,
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "model_assignments": [a.to_dict() for a in self.model_assignments],
        }
        # Only written when present, so jobs.yaml entries without a
        # camera location stay exactly as they were.
        if self.latitude is not None:
            result["latitude"] = self.latitude
        if self.longitude is not None:
            result["longitude"] = self.longitude
        return result

    @staticmethod
    def from_dict(raw: dict) -> "JobConfig":
        assignments_raw = raw.get("model_assignments") or []
        return JobConfig(
            job_id=str(raw["job_id"]),
            branch_id=str(raw["branch_id"]),
            branch_name=str(raw["branch_name"]),
            camera_id=str(raw["camera_id"]),
            camera_name=str(raw["camera_name"]),
            camera_stream=str(raw["camera_stream"]),
            zone_id=str(raw["zone_id"]),
            zone_name=str(raw["zone_name"]),
            model_assignments=tuple(ModelAssignment.from_dict(a) for a in assignments_raw),
            latitude=_optional_float(raw.get("latitude")),
            longitude=_optional_float(raw.get("longitude")),
        )

    def config_changed(self, other: "JobConfig") -> bool:
        """True if anything other than job_id differs — used by
        job_watcher.py to classify CHANGED vs UNCHANGED.
        """
        return self.to_dict() != other.to_dict()
