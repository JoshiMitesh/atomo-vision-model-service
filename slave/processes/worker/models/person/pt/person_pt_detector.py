"""Person `.pt` detector — Ultralytics/PyTorch inference path.

Responsible for preprocessing, inference, decoding, confidence
filtering and NMS (delegated to Ultralytics), and coordinate
conversion into the shared Detection contract. No RTSP, MinIO, or NATS
code belongs here (AGENTS.md §7, §18).
"""

from __future__ import annotations

from typing import List

import numpy as np

from slave.models.detection import Detection
from slave.runtimes.ultralytics.ultralytics_runtime import UltralyticsRuntime

CLASS_ID = 0
CLASS_NAME = "person"


class PersonPtDetector:
    def __init__(self, model_path: str, model_config: dict) -> None:
        pt_cfg = model_config.get("pt", {})
        thresholds = model_config.get("thresholds", {})
        self._obj_thresh = float(thresholds.get("obj_thresh", 0.20))
        self._nms_thresh = float(thresholds.get("nms_thresh", 0.65))
        self._runtime = UltralyticsRuntime(
            model_path=model_path,
            device=pt_cfg.get("device", "auto"),
            half=bool(pt_cfg.get("half", False)),
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        result = self._runtime.infer(frame, conf=self._obj_thresh, iou=self._nms_thresh)
        detections: List[Detection] = []

        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id != CLASS_ID:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            detections.append(
                Detection(
                    class_id=CLASS_ID,
                    class_name=CLASS_NAME,
                    confidence=float(box.conf[0]),
                    bbox=(x1, y1, x2, y2),
                )
            )
        return detections

    def release(self) -> None:
        self._runtime.release()
