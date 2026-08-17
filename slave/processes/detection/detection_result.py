"""DetectionResult — the normalized output of every Person detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from slave.models.detection import Detection


@dataclass(frozen=True)
class DetectionResult:
    job_id: str
    frame_id: int
    detections: List[Detection]

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "frame_id": self.frame_id,
            "detections": [d.to_dict() for d in self.detections],
        }

    @property
    def has_detections(self) -> bool:
        return len(self.detections) > 0
