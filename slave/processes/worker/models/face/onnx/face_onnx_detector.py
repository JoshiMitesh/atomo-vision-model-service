"""TEMPORARY TEST detector — runs a raw .onnx face-detection model
(YuNet, via OpenCV's cv2.FaceDetectorYN, same loader face_embedder.py
already uses for landmarks) as the job's actual face detector.

NOT part of the permanent .pt/.nb/.so architecture (CLAUDE.md §7/§22
whitelists only those three extensions). This exists so a downloaded
.onnx file can be tried against a live job without touching the real
pt/nb/so pipeline. Delete this folder, the model_registry.py/
model_resolver.py .onnx entries, and revert jobs.yaml's model_name
once testing is done.

Unlike face_pt_detector.py (Ultralytics/YOLO output contract), YuNet's
own ONNX has a completely different output format that
UltralyticsRuntime can't parse — hence this separate loader instead of
reusing UltralyticsRuntime.
"""

from __future__ import annotations

from typing import List

import cv2 as cv
import numpy as np

from slave.models.detection import Detection

CLASS_ID = 1
CLASS_NAME = "face"


class FaceOnnxDetector:
    def __init__(self, model_path: str, model_config: dict, model_name: str) -> None:
        self._model_name = model_name
        thresholds = model_config.get("thresholds", {})
        self._obj_thresh = float(thresholds.get("obj_thresh", 0.60))
        self._nms_thresh = float(thresholds.get("nms_thresh", 0.30))
        self._model_path = model_path
        self._detector = None  # built lazily once the first frame's size is known
        self._input_size = None

    def _ensure_detector(self, width: int, height: int) -> None:
        if self._detector is not None and self._input_size == (width, height):
            return
        self._detector = cv.FaceDetectorYN.create(
            self._model_path, "", (width, height),
            score_threshold=self._obj_thresh, nms_threshold=self._nms_thresh, top_k=5000,
        )
        self._input_size = (width, height)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        self._ensure_detector(w, h)
        _, faces = self._detector.detect(frame)

        detections: List[Detection] = []
        if faces is None:
            return detections

        for f in faces:
            x, y, fw, fh = f[:4]
            conf = float(f[-1])
            detections.append(
                Detection(
                    class_id=CLASS_ID,
                    class_name=CLASS_NAME,
                    confidence=conf,
                    bbox=(float(x), float(y), float(x + fw), float(y + fh)),
                    model_name=self._model_name,
                )
            )
        return detections

    def release(self) -> None:
        self._detector = None
