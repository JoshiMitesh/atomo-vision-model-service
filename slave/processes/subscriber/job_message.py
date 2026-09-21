"""Parse/validate an incoming job message into a JobConfig.

Expected shape (Master -> Slave, slave.jobs.start):
    {
      "job_id": "job_001",
      "camera_id": "773c74ea-4767-40db-92de-9574c97991ec",
      "zone_id": "main-entrance",
      "model_assignments": [
        {"model_type": "person", "model_name": "person_yolo26n.pt", "camera_model_assign_id": "032730bb-..."},
        {"model_type": "face",   "model_name": "face.pt",           "camera_model_assign_id": "77614d82-..."}
      ]
    }

There is no top-level model_name/model_type/model_extension anymore —
each model lives in its own model_assignments[] entry (models/job.py's
JobConfig.primary_assignment picks which one drives the running
worker). master_ip/stream_port/stream_name are gone too — the Slave is
already NATS-connected to Master, and the stream is resolved from
camera_id alone.
"""

from __future__ import annotations

from slave.models.job import JobConfig

REQUIRED_FIELDS = (
    "job_id",
    "camera_id",
    "zone_id",
    "model_assignments",
)

REQUIRED_ASSIGNMENT_FIELDS = ("model_type", "model_name", "camera_model_assign_id")


class InvalidJobMessage(ValueError):
    pass


def parse_job_message(raw: dict) -> JobConfig:
    if not isinstance(raw, dict):
        raise InvalidJobMessage("job message must be a JSON object")

    missing = [f for f in REQUIRED_FIELDS if f not in raw]
    if missing:
        raise InvalidJobMessage(f"job message missing fields: {missing}")

    assignments = raw.get("model_assignments")
    if not isinstance(assignments, list) or not assignments:
        raise InvalidJobMessage("job message model_assignments must be a non-empty list")
    for i, assignment in enumerate(assignments):
        if not isinstance(assignment, dict):
            raise InvalidJobMessage(f"job message model_assignments[{i}] must be an object")
        missing_fields = [f for f in REQUIRED_ASSIGNMENT_FIELDS if f not in assignment]
        if missing_fields:
            raise InvalidJobMessage(f"job message model_assignments[{i}] missing fields: {missing_fields}")

    try:
        return JobConfig.from_dict(raw)
    except (TypeError, ValueError, KeyError) as exc:
        raise InvalidJobMessage(f"job message has invalid field types: {exc}") from exc
