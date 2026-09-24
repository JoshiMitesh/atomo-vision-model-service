"""Fire detection severity -- derived from bbox size relative to frame
area (bigger box = closer/larger fire = higher severity), not a model
output. Thresholds come from config/models/fire.yaml's
severity_thresholds: block. Shared by the pt/nb/so fire workers.
"""

from __future__ import annotations

from typing import Tuple

DEFAULT_LOW_MAX = 0.05
DEFAULT_MEDIUM_MAX = 0.15


def compute_severity(
    bbox: Tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
    model_config: dict,
) -> str:
    frame_area = float(frame_width) * float(frame_height)
    if frame_area <= 0:
        return "low"

    x1, y1, x2, y2 = bbox
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ratio = box_area / frame_area

    thresholds = model_config.get("severity_thresholds") or {}
    low_max = thresholds.get("low_max", DEFAULT_LOW_MAX)
    medium_max = thresholds.get("medium_max", DEFAULT_MEDIUM_MAX)

    if ratio < low_max:
        return "low"
    if ratio < medium_max:
        return "medium"
    return "high"
