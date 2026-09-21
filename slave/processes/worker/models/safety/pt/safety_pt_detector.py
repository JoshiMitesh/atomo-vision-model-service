"""Safety `.pt` detector — Ultralytics/PyTorch inference path.

Mirrors person_pt_detector.py's structure, but Safety is multi-class
(9 classes: boots, gloves, hardhat, no_boots, no_gloves, no_hardhat,
no_vest, person, vest) — unlike Person/Face, every class Ultralytics
returns is kept and mapped to its own class_id/class_name, not
filtered/remapped to one fixed class (CLAUDE.md §9's "do not build a
generic multi-class decoder and assume it will always match" concerns
the NPU DFL decoder, not this Ultralytics-native path — Ultralytics
already gives per-box class ids for free here).
"""

from __future__ import annotations

from typing import List

import numpy as np

from slave.models.detection import Detection
from slave.runtimes.ultralytics.ultralytics_runtime import UltralyticsRuntime

CLASSES = (
    "boots",
    "gloves",
    "hardhat",
    "no_boots",
    "no_gloves",
    "no_hardhat",
    "no_vest",
    "person",
    "vest",
)


class SafetyPtDetector:
    def __init__(self, model_path: str, model_config: dict, model_name: str) -> None:
        self._model_name = model_name
        # Published Detection.model_name — the ENUM/display name (e.g.
        # "Safety"), not the job's model filename. See config/models/
        # person.yaml's `display_name` comment for why.
        self._display_name = model_config.get("display_name") or model_name
        pt_cfg = model_config.get("pt", {})
        thresholds = model_config.get("thresholds", {})
        self._obj_thresh = float(thresholds.get("obj_thresh", 0.25))
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

        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id < 0 or cls_id >= len(CLASSES):
                continue  # defensive — should never happen against this checkpoint
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=CLASSES[cls_id],
                    confidence=float(box.conf[0]),
                    bbox=(x1, y1, x2, y2),
                    model_name=self._display_name,
                )
            )
        return detections

    def release(self) -> None:
        self._runtime.release()
