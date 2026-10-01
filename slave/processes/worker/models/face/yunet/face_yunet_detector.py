"""Face YuNet detector — job-selected ONNX YuNet model.

The YuNet model filename comes from jobs.yaml via ModelAssignment.model_name.
SFace is intentionally not a job model; it remains an internal dependency of
embedding/face_embedder.py.
"""

from __future__ import annotations

from typing import List

import cv2 as cv
import numpy as np

from slave.models.detection import Detection

CLASS_ID = 1
CLASS_NAME = "face"


class FaceYunetDetector:
    def __init__(self, model_path: str, model_config: dict, model_name: str) -> None:
        self._model_name = model_name
        cfg = model_config.get("yunet", {})
        thresholds = model_config.get("thresholds", {})

        self._score_thresh = float(cfg.get("score_thresh", thresholds.get("obj_thresh", 0.60)))
        self._nms_thresh = float(cfg.get("nms_thresh", thresholds.get("nms_thresh", 0.50)))
        self._top_k = int(cfg.get("top_k", 5000))

        self._detector = cv.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            self._score_thresh,
            self._nms_thresh,
            self._top_k,
            cv.dnn.DNN_BACKEND_OPENCV,
            cv.dnn.DNN_TARGET_CPU,
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        height, width = frame.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(frame)

        if faces is None:
            return []

        detections: List[Detection] = []
        for face in np.asarray(faces, dtype=np.float32):
            if face.shape[0] < 15:
                continue

            x, y, w, h = (float(v) for v in face[:4])
            confidence = float(face[-1])

            detections.append(
                Detection(
                    class_id=CLASS_ID,
                    class_name=CLASS_NAME,
                    confidence=confidence,
                    bbox=(x, y, x + w, y + h),
                    model_name=self._model_name,
                    extra={"yunet_landmarks": face[4:14].tolist()},
                )
            )

        return detections

    def release(self) -> None:
        self._detector = None
