"""Face `.pt` detector — Ultralytics/PyTorch inference path.

Mirrors person_pt_detector.py's structure exactly (reuses the same
generic UltralyticsRuntime adapter); only the class identity differs.
Only ever runs while FaceController has it activated — i.e. while the
Person detector on the same job is seeing a person, or within the
post-person grace window (processes/worker/face/face_controller.py).
"""

from __future__ import annotations

from typing import List

import numpy as np

from slave.models.detection import Detection
from slave.runtimes.ultralytics.ultralytics_runtime import UltralyticsRuntime

CLASS_ID = 1
CLASS_NAME = "face"


class FacePtDetector:
    def __init__(self, model_path: str, model_config: dict, model_name: str) -> None:
        self._model_name = model_name
        pt_cfg = model_config.get("pt", {})
        thresholds = model_config.get("thresholds", {})
        self._obj_thresh = float(thresholds.get("obj_thresh", 0.35))
        self._nms_thresh = float(thresholds.get("nms_thresh", 0.45))
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

        # Dedicated single-class face model (mirrors the Person .pt
        # path's NUM_CLS=1 assumption, CLAUDE.md §9) — every box
        # Ultralytics returns is a face, regardless of the internal
        # class index it assigns (typically 0), so it's remapped to
        # this project's own face class identity here.
        for box in boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            detections.append(
                Detection(
                    class_id=CLASS_ID,
                    class_name=CLASS_NAME,
                    confidence=float(box.conf[0]),
                    bbox=(x1, y1, x2, y2),
                    model_name=self._model_name,
                )
            )
        return detections

    def release(self) -> None:
        self._runtime.release()
