"""DetectionResult — the normalized cam.detection payload for one frame.

`telemetry` is a list of ClassDetection (models/detection.py) — one
entry per class_name (person, face, ...), each grouping every instance
of that class detected in this frame. `object_key`/`url` are set only
when this frame was uploaded to MinIO (i.e. it had detections); the
same value doubles as every ClassDetection's `tracking_id` (see
models/detection.py's group_detections — tracking_id is just the
frame's saved MinIO image id, not a real tracker, per product
decision).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from slave.models.detection import ClassDetection

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DetectionResult:
    job_id: str
    frame_id: int
    occurred_at: str

    branch_id: str
    device_id: str
    zone_id: str
    camera_id: str
    camera_status: str

    telemetry: List[ClassDetection]

    object_key: Optional[str] = None
    url: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "job_id": self.job_id,
            "frame_id": self.frame_id,
            "occurred_at": self.occurred_at,
            "source": {
                "branch_id": self.branch_id,
                "device_id": self.device_id,
                "zone_id": self.zone_id,
                "camera": {
                    "camera_id": self.camera_id,
                    "status": self.camera_status,
                },
            },
            "media": {
                "object_key": self.object_key,
                "url": self.url,
                "width": self.width,
                "height": self.height,
            },
            "telemetry": [t.to_dict() for t in self.telemetry],
        }

    @property
    def has_detections(self) -> bool:
        return any(t.instances for t in self.telemetry)
