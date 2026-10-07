"""Face enrollment processor using the exact production YuNet + SFace pipeline.

Enrollment is intentionally separate from live face detection. It accepts one
small JPEG frame at a time, extracts the best complete face, and returns the
same 128-D SFace embedding used by live recognition.
"""

from __future__ import annotations

import base64
import os
from typing import Optional

import cv2 as cv
import numpy as np

from slave.processes.worker.models.face.embedding.face_image_enhancer import enhance_face


class FaceEnrollmentProcessor:
    def __init__(self, models_root: str) -> None:
        face_dir = os.path.join(models_root, "face")
        self._recognizer = cv.FaceRecognizerSF.create(
            os.path.join(face_dir, "face_recognition_sface_2021dec_int8.onnx"),
            "",
            cv.dnn.DNN_BACKEND_OPENCV,
            cv.dnn.DNN_TARGET_CPU,
        )
        self._landmarker = cv.FaceDetectorYN.create(
            os.path.join(face_dir, "face_detection_yunet_2023mar_int8.onnx"),
            "",
            (320, 320),
            0.80,
            0.30,
            5000,
            cv.dnn.DNN_BACKEND_OPENCV,
            cv.dnn.DNN_TARGET_CPU,
        )

    def process(self, frame_base64: str) -> dict:
        try:
            if "," in frame_base64:
                frame_base64 = frame_base64.split(",", 1)[1]
            raw = base64.b64decode(frame_base64, validate=True)
            frame = cv.imdecode(np.frombuffer(raw, dtype=np.uint8), cv.IMREAD_COLOR)
            if frame is None:
                return {"ok": False, "reason": "invalid_image"}

            height, width = frame.shape[:2]
            self._landmarker.setInputSize((width, height))
            _, faces = self._landmarker.detect(frame)
            if faces is None or len(faces) == 0:
                return {"ok": False, "reason": "no_face"}

            candidates = []
            for face in np.asarray(faces, dtype=np.float32):
                if face.shape[0] < 15 or not np.isfinite(face).all():
                    continue
                confidence = float(face[-1])
                if confidence < 0.75:
                    continue
                x, y, w, h = map(float, face[:4])
                if w < 50.0 or h < 50.0:
                    continue
                bbox = (x, y, x + w, y + h)
                if not self._landmarks_look_complete(face, bbox):
                    continue
                candidates.append((w * h, confidence, face))

            if not candidates:
                return {"ok": False, "reason": "face_quality"}

            _, confidence, face = max(candidates, key=lambda item: item[0])
            aligned = self._recognizer.alignCrop(
                frame,
                face[:-1].astype(np.float32),
            )
            aligned = enhance_face(aligned)
            feature = self._recognizer.feature(aligned)
            if feature is None or feature.size == 0:
                return {"ok": False, "reason": "embedding_failed"}

            vector = np.asarray(feature, dtype=np.float32).flatten()
            if not np.isfinite(vector).all():
                return {"ok": False, "reason": "embedding_failed"}

            # Store the same raw SFace vector format as live recognition.
            embedding = [round(float(value), 6) for value in vector]

            # Return the same-size enhanced enrollment crop so the
            # visible profile photo also benefits from the cleanup.
            aligned = self._recognizer.alignCrop(
                frame,
                face[:-1].astype(np.float32),
            )
            aligned = enhance_face(aligned)
            ok, encoded = cv.imencode(
                ".jpg",
                aligned,
                [int(cv.IMWRITE_JPEG_QUALITY), 92],
            )
            if not ok:
                return {"ok": False, "reason": "photo_encode_failed"}

            return {
                "ok": True,
                "embedding": embedding,
                "photo": base64.b64encode(encoded.tobytes()).decode("ascii"),
                "confidence": confidence,
                "width": int(face[2]),
                "height": int(face[3]),
            }
        except Exception as exc:
            return {"ok": False, "reason": f"processor_error:{type(exc).__name__}"}

    @staticmethod
    def _landmarks_look_complete(
        landmark_face: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> bool:
        if landmark_face is None or len(landmark_face) < 14:
            return False

        points = np.asarray(
            landmark_face[4:14],
            dtype=np.float32,
        ).reshape(5, 2)
        if not np.isfinite(points).all():
            return False

        x1, y1, x2, y2 = map(float, bbox)
        width = x2 - x1
        height = y2 - y1
        if width < 1.0 or height < 1.0:
            return False

        margin_x = 0.05 * width
        margin_y = 0.05 * height
        if np.any(points[:, 0] < x1 - margin_x) or np.any(points[:, 0] > x2 + margin_x):
            return False
        if np.any(points[:, 1] < y1 - margin_y) or np.any(points[:, 1] > y2 + margin_y):
            return False

        right_eye, left_eye, nose, right_mouth, left_mouth = points
        eye_y = (right_eye[1] + left_eye[1]) * 0.5
        mouth_y = (right_mouth[1] + left_mouth[1]) * 0.5

        if nose[1] <= eye_y:
            return False
        if mouth_y <= eye_y + (0.12 * height):
            return False
        if mouth_y <= nose[1] + (0.03 * height):
            return False

        eye_gap = float(np.linalg.norm(right_eye - left_eye))
        mouth_gap = float(np.linalg.norm(right_mouth - left_mouth))
        return eye_gap >= 0.08 * width and mouth_gap >= 0.05 * width
