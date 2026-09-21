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
paths, and plays no part in connecting to the stream.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ModelAssignment:
    """One model Master has assigned to this job's camera."""

    model_type: str
    model_name: str
    camera_model_assign_id: str

    @property
    def model_extension(self) -> str:
        return os.path.splitext(self.model_name)[1]

    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "camera_model_assign_id": self.camera_model_assign_id,
        }

    @staticmethod
    def from_dict(raw: dict) -> "ModelAssignment":
        return ModelAssignment(
            model_type=str(raw["model_type"]),
            model_name=str(raw["model_name"]),
            camera_model_assign_id=str(raw["camera_model_assign_id"]),
        )


@dataclass(frozen=True)
class JobConfig:
    job_id: str
    camera_id: str
    camera_stream: str
    zone_id: str
    model_assignments: Tuple[ModelAssignment, ...]

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

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "camera_id": self.camera_id,
            "camera_stream": self.camera_stream,
            "zone_id": self.zone_id,
            "model_assignments": [a.to_dict() for a in self.model_assignments],
        }

    @staticmethod
    def from_dict(raw: dict) -> "JobConfig":
        assignments_raw = raw.get("model_assignments") or []
        return JobConfig(
            job_id=str(raw["job_id"]),
            camera_id=str(raw["camera_id"]),
            camera_stream=str(raw["camera_stream"]),
            zone_id=str(raw["zone_id"]),
            model_assignments=tuple(ModelAssignment.from_dict(a) for a in assignments_raw),
        )

    def config_changed(self, other: "JobConfig") -> bool:
        """True if anything other than job_id differs — used by
        job_watcher.py to classify CHANGED vs UNCHANGED.
        """
        return self.to_dict() != other.to_dict()
