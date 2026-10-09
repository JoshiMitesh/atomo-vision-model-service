"""Number-plate `.pt` detector -- Ultralytics/PyTorch inference path.

Single-class plate detector (yolov11s fine-tuned on license plates;
confirmed via the checkpoint's own `names` dict: {0: 'License_Plate'}).
Mirrors fire_pt_detector.py's structure. Detection (box + confidence +
shape filter) only -- OCR and cross-frame plate-text voting are a
separate, per-job stateful concern (plate_enricher.py, invoked by the
worker) and do not belong in the detector per CLAUDE.md SS8.

The size/aspect-ratio/area shape filter is ported from the updated
reference's plate_bbox_valid() (alpr_rtsp.py) -- live RTSP testing
showed the raw YOLO output includes square-ish false positives (signs,
faces, etc.) that no plate is ever shaped like; rejecting those here
means they never reach tracking/OCR at all.
"""

from __future__ import annotations

from typing import List

import numpy as np

from slave.models.detection import Detection
from slave.runtimes.ultralytics.ultralytics_runtime import UltralyticsRuntime

CLASS_NAME = "number_plate"


class NumberPlatePtDetector:
    def __init__(self, model_path: str, model_config: dict, model_name: str) -> None:
        # Published Detection.model_name -- the ENUM/display name (e.g.
        # "Number Plate"), not the job's model filename. See config/
        # models/person.yaml's `display_name` comment for why.
        self._display_name = model_config.get("display_name") or model_name

        thresholds = model_config.get("thresholds", {})
        self._obj_thresh = float(thresholds.get("obj_thresh", 0.55))
        self._nms_thresh = float(thresholds.get("nms_thresh", 0.45))

        filters = model_config.get("pt", {}).get("filters", model_config.get("filters", {}))
        self._min_w = float(filters.get("min_plate_width_px", 45))
        self._min_h = float(filters.get("min_plate_height_px", 15))
        self._min_aspect = float(filters.get("min_plate_aspect", 2.0))
        self._max_aspect = float(filters.get("max_plate_aspect", 8.0))
        self._min_area_ratio = float(filters.get("min_plate_area_ratio", 0.00005))

        pt_cfg = model_config.get("pt", {})
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

        img_h, img_w = frame.shape[:2]
        image_area = float(img_w * img_h)

        for box in boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            if not self._plate_shape_valid(x1, y1, x2, y2, image_area):
                continue
            detections.append(
                Detection(
                    class_id=0,
                    class_name=CLASS_NAME,
                    confidence=float(box.conf[0]),
                    bbox=(x1, y1, x2, y2),
                    model_name=self._display_name,
                )
            )
        return detections

    def _plate_shape_valid(self, x1: float, y1: float, x2: float, y2: float, image_area: float) -> bool:
        w, h = x2 - x1, y2 - y1
        if w < self._min_w or h <= 0 or h < self._min_h:
            return False
        aspect = w / h
        if aspect < self._min_aspect or aspect > self._max_aspect:
            return False
        if (w * h) < image_area * self._min_area_ratio:
            return False
        return True

    def release(self) -> None:
        self._runtime.release()
