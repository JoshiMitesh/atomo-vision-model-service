"""SFace face embedder — one 128-d embedding per detected face.

Runs *after* a Face detector (pt/nb/so) has produced its boxes; it
never detects faces itself (detection stays in the detectors per
CLAUDE.md §8). Pipeline per face, matching the reference
face_worker.py's YuNet -> SFace.alignCrop -> SFace.feature() flow:

    bbox -> padded crop -> YuNet (5 landmarks, on the small crop only)
         -> SFace.alignCrop (aligned 112x112) -> SFace.feature()
         -> float list (JSON-safe)

The face detectors here output only a bbox, no landmarks, so YuNet is
run on each padded face crop purely to recover the 5 landmarks SFace's
alignment needs. There is deliberately NO unaligned fallback: measured
against the reference pipeline, a plain bbox crop resized to 112x112
gives cosine <= ~0.57 to the aligned embedding (aligned-from-bbox: ~0.95),
so it would silently fail to match anything the Master enrolled. If
landmarks cannot be found for a face, that face's embedding is `None`.

Any failure yields `None` for that face's embedding — it must never
drop or delay the detection itself. One FaceEmbedder per job, never
shared (CLAUDE.md §20).
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np

from slave.logging.logger import get_logger
from slave.models.detection import Detection

log = get_logger(__name__)

_LANDMARK_CROP_PAD = 0.25  # padding around the bbox before running YuNet on it
_LANDMARK_CROP_SIDE = 192  # crop's longer side is scaled to this for YuNet
_LANDMARK_SCORE_THRESH = 0.5
_LANDMARK_NMS_THRESH = 0.3
_LANDMARK_TOP_K = 50


class FaceEmbedder:
    def __init__(self, embedding_config: dict, models_dir: str) -> None:
        self._model_name = str(embedding_config.get("model_name", "sface_2021dec_int8"))
        self._min_face_px = int(embedding_config.get("min_face_px", 20))
        self._min_det_conf = float(embedding_config.get("min_det_conf", 0.5))
        self._max_faces = int(embedding_config.get("max_faces_per_frame", 10))
        self._l2_normalize = bool(embedding_config.get("l2_normalize", False))
        self._decimals = int(embedding_config.get("decimals", 6))

        self._recognizer = None
        self._landmarker = None

        model_file = str(embedding_config.get("model_file", "face_recognition_sface_2021dec_int8.onnx"))
        model_path = os.path.join(models_dir, model_file)
        try:
            self._recognizer = cv.FaceRecognizerSF.create(model_path, "", cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU)
        except Exception:
            log.exception("SFace model failed to load from %s — face embeddings disabled", model_path)
            return

        landmark_file = str(embedding_config.get("landmark_model_file", "face_detection_yunet_2023mar_int8.onnx"))
        landmark_path = os.path.join(models_dir, landmark_file)
        try:
            self._landmarker = cv.FaceDetectorYN.create(
                landmark_path, "", (_LANDMARK_CROP_SIDE, _LANDMARK_CROP_SIDE),
                _LANDMARK_SCORE_THRESH, _LANDMARK_NMS_THRESH, _LANDMARK_TOP_K,
                cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU,
            )
        except Exception:
            log.exception("YuNet landmark model failed to load from %s — face embeddings disabled", landmark_path)
            self._recognizer = None
            return

        log.info("face embedder ready (model=%s)", self._model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def available(self) -> bool:
        return self._recognizer is not None

    def attach(self, frame: np.ndarray, detections: List[Detection]) -> List[Detection]:
        """Return `detections` with `embedding` / `embedding_model` set in
        each face's `extra`. Faces that are skipped (too small, low
        confidence, over the per-frame cap) or fail to embed keep
        `embedding: None` so the JSON shape stays identical.
        """
        if not detections:
            return detections
        if not self.available:
            return [self._with_embedding(d, None) for d in detections]

        # Largest faces first; only the top `max_faces` get an embedding.
        order = sorted(range(len(detections)), key=lambda i: self._area(detections[i].bbox), reverse=True)
        embeddable = set(order[: self._max_faces])

        result: List[Detection] = []
        for idx, det in enumerate(detections):
            embedding = None
            if idx in embeddable and self._is_embeddable(det):
                embedding = self._embed(frame, det.bbox)
            result.append(self._with_embedding(det, embedding))
        return result

    def release(self) -> None:
        self._recognizer = None
        self._landmarker = None

    def _with_embedding(self, det: Detection, embedding: Optional[List[float]]) -> Detection:
        return replace(
            det,
            extra={**det.extra, "embedding": embedding, "embedding_model": self._model_name if embedding else None},
        )

    def _is_embeddable(self, det: Detection) -> bool:
        x1, y1, x2, y2 = det.bbox
        return (
            det.confidence >= self._min_det_conf
            and (x2 - x1) >= self._min_face_px
            and (y2 - y1) >= self._min_face_px
        )

    @staticmethod
    def _area(bbox: Tuple[float, float, float, float]) -> float:
        return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])

    def _embed(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[List[float]]:
        try:
            aligned = self._align_with_landmarks(frame, bbox)
            if aligned is None:
                return None

            feature = self._recognizer.feature(aligned)
            if feature is None or feature.size == 0:
                return None
            vector = np.asarray(feature, dtype=np.float32).flatten()
            if not np.isfinite(vector).all():
                return None
            if self._l2_normalize:
                norm = float(np.linalg.norm(vector))
                if norm == 0.0:
                    return None
                vector = vector / norm
            return [round(float(v), self._decimals) for v in vector]
        except Exception:
            log.exception("face embedding failed for bbox=%s", bbox)
            return None

    def _padded_crop(
        self, frame: np.ndarray, bbox: Tuple[float, float, float, float], pad: float
    ) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        pad_x, pad_y = (x2 - x1) * pad, (y2 - y1) * pad
        cx1, cy1 = max(0, int(round(x1 - pad_x))), max(0, int(round(y1 - pad_y)))
        cx2, cy2 = min(w, int(round(x2 + pad_x))), min(h, int(round(y2 + pad_y)))
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        return frame[cy1:cy2, cx1:cx2]

    def _align_with_landmarks(
        self, frame: np.ndarray, bbox: Tuple[float, float, float, float]
    ) -> Optional[np.ndarray]:
        crop = self._padded_crop(frame, bbox, _LANDMARK_CROP_PAD)
        if crop is None:
            return None

        # Scale so the crop's longer side is _LANDMARK_CROP_SIDE: small faces
        # are upscaled for YuNet, large ones downscaled to bound its cost.
        ch, cw = crop.shape[:2]
        scale = _LANDMARK_CROP_SIDE / float(max(ch, cw))
        crop = cv.resize(
            crop, (max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))), interpolation=cv.INTER_LINEAR
        )
        ch, cw = crop.shape[:2]

        self._landmarker.setInputSize((cw, ch))
        _, faces = self._landmarker.detect(crop)
        if faces is None or len(faces) == 0:
            return None

        # The crop holds one intended face; if YuNet finds several (a
        # neighbour at the crop edge), take the largest.
        best = max(faces, key=lambda f: float(f[2]) * float(f[3]))
        aligned = self._recognizer.alignCrop(crop, best[:-1].astype(np.float32))
        if aligned is None or aligned.size == 0:
            return None
        return aligned
