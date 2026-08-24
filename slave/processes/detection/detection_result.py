"""DetectionResult — the normalized output published for one frame.

`detections` may now be a merged list from more than one model (Person
+ Face). There is deliberately no top-level `model_name` here anymore —
each `Detection` carries its own (see models/detection.py); a single
result can no longer be described by one model name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from slave.models.detection import Detection


@dataclass(frozen=True)
class DetectionResult:
    job_id: str
    frame_id: int
    camera_name: str
    camera_url: str
    camera_status: str
    detections: List[Detection]

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "frame_id": self.frame_id,
            "camera": {
                "camera_name": self.camera_name,
                "camera_url": self.camera_url,
                "camera_status": self.camera_status,
            },
            "detections": [d.to_dict() for d in self.detections],
        }

    @property
    def has_detections(self) -> bool:
        return len(self.detections) > 0
