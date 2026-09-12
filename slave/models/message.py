"""Generic NATS message envelope helpers.

The specific parsers (processes/subscriber/job_message.py,
processes/subscriber/stop_message.py,
processes/detection/detection_message.py) build on top of this — this
module only knows how to classify a raw payload, not how to validate
one particular message type.
"""

from __future__ import annotations

from enum import Enum


class MessageType(str, Enum):
    JOB = "job"
    STOP = "stop"
    UNKNOWN = "unknown"


def classify(raw: dict) -> MessageType:
    """Classify a raw decoded NATS payload by shape.

    A stop message is ``{"action": "stop", "job_id": ...}``.
    A job message carries the full job descriptor (no "action" field) —
    identified by job_id + model_assignments (there is no top-level
    model_type anymore, see models/job.py).
    """
    if not isinstance(raw, dict):
        return MessageType.UNKNOWN
    if raw.get("action") == "stop":
        return MessageType.STOP
    if "model_assignments" in raw and "job_id" in raw:
        return MessageType.JOB
    return MessageType.UNKNOWN
