"""DetectionResult — the normalized output published for one frame.

`detections` may now be a merged list from more than one model (Person
+ Face). There is deliberately no top-level `model_name` here anymore —
each `Detection` carries its own (see models/detection.py); a single
result can no longer be described by one model name.

`object_key`/`url` are set only when this frame was uploaded to MinIO
(i.e. it had detections) — the frame-upload NATS message is merged
into this one instead of being published separately (was previously a
second message via UploadPublisher; see person_*_worker.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from slave.models.detection import Detection


@dataclass(frozen=True)
class DetectionResult:
    job_id: str
    frame_id: int
    camera_name: str
    camera_url: str
    camera_status: str
    camera_fps: float
    detections: List[Detection]
    object_key: Optional[str] = None
    url: Optional[str] = None

    def to_dict(self) -> dict:
        result: dict = {
            "job_id": self.job_id,
            "frame_id": self.frame_id,
        }
        if self.object_key is not None:
            result["object_key"] = self.object_key
        if self.url is not None:
            result["url"] = self.url
        result["camera"] = {
            "camera_name": self.camera_name,
            "camera_url": self.camera_url,
            "camera_status": self.camera_status,
        }
        result["detections"] = [d.to_dict() for d in self.detections]
        return result

    @property
    def has_detections(self) -> bool:
        return len(self.detections) > 0
