"""Parse/validate an incoming job message into a JobConfig.

Expected shape (see AGENTS.md §4):
    {
      "job_id": "job_001",
      "model_name": "person_yolo26n.pt",
      "model_type": "person",
      "model_extension": ".pt",
      "master_ip": "192.168.1.10",
      "stream_port": 8554,
      "stream_name": "camera1"
    }
"""

from __future__ import annotations

from slave.models.job import JobConfig

REQUIRED_FIELDS = (
    "job_id",
    "model_name",
    "model_type",
    "model_extension",
    "master_ip",
    "stream_port",
    "stream_name",
)


class InvalidJobMessage(ValueError):
    pass


def parse_job_message(raw: dict) -> JobConfig:
    if not isinstance(raw, dict):
        raise InvalidJobMessage("job message must be a JSON object")

    missing = [f for f in REQUIRED_FIELDS if f not in raw]
    if missing:
        raise InvalidJobMessage(f"job message missing fields: {missing}")

    try:
        return JobConfig.from_dict(raw)
    except (TypeError, ValueError) as exc:
        raise InvalidJobMessage(f"job message has invalid field types: {exc}") from exc
