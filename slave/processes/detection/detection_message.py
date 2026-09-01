"""Build the JSON-safe NATS payload for a detection result."""

from __future__ import annotations

from slave.processes.detection.detection_result import DetectionResult
from slave.utils.json_utils import to_json_safe


def build_detection_message(result: DetectionResult) -> dict:
    """CLAUDE.md §16: must be JSON-serializable, no NumPy objects."""
    return to_json_safe(result.to_dict())
