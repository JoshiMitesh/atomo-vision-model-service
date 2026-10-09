"""SFace face embedder — one 128-d embedding per detected face.

The face detector (pt/nb/so) supplies the detection boxes.  For SFace
alignment, YuNet is run on the ORIGINAL full frame, matching the reference
atomo-ai-mitesh pipeline:

    full frame -> YuNet (5 landmarks) -> SFace.alignCrop(full frame, landmarks)
               -> SFace.feature() -> 128-float embedding

The existing detector remains responsible for the published face boxes.
YuNet is used only to recover the landmarks needed by SFace.  The YuNet
detection corresponding to each detector bbox is selected using bbox overlap
and centre distance so crowded scenes do not accidentally use a neighbour's
landmarks.

Any embedding failure returns None and never removes/delays the detection.
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

# These match the reference YuNet configuration.
_LANDMARK_SCORE_THRESH = 0.70
_LANDMARK_NMS_THRESH = 0.30
_LANDMARK_TOP_K = 5000


class FaceEmbedder:
    def __init__(
        self,
        embedding_config: dict,
        models_dir: str,
        yunet_model_name: Optional[str] = None,
    ) -> None:
        self._model_name = str(
            embedding_config.get("model_name", "face_recognition")
        )
        self._min_face_px = int(embedding_config.get("min_face_px", 20))
        self._min_det_conf = float(embedding_config.get("min_det_conf", 0.5))
        self._max_faces = int(embedding_config.get("max_faces_per_frame", 10))
        self._l2_normalize = bool(embedding_config.get("l2_normalize", False))
        self._decimals = int(embedding_config.get("decimals", 6))

        self._landmark_score_thresh = float(
            embedding_config.get("landmark_score_thresh", _LANDMARK_SCORE_THRESH)
        )
        self._landmark_nms_thresh = float(
            embedding_config.get("landmark_nms_thresh", _LANDMARK_NMS_THRESH)
        )
        self._landmark_top_k = int(
            embedding_config.get("landmark_top_k", _LANDMARK_TOP_K)
        )

        self._recognizer = None
        self._landmarker = None

        # SFace is intentionally an internal implementation dependency.
        # It is never selected from jobs.yaml.
        model_file = "face_recognition_sface_2021dec_int8.onnx"
        model_path = os.path.join(models_dir, model_file)
        try:
            self._recognizer = cv.FaceRecognizerSF.create(
                model_path,
                "",
                cv.dnn.DNN_BACKEND_OPENCV,
                cv.dnn.DNN_TARGET_CPU,
            )
        except Exception:
            log.exception(
                "SFace model failed to load from %s — face embeddings disabled",
                model_path,
            )
            return

        # YuNet is normally selected by the face job's model_assignment.
        # Keep the old filename as a backward-compatible fallback for
        # Person-triggered FaceController jobs that do not carry a YuNet
        # assignment yet.
        landmark_file = str(
            yunet_model_name
            or embedding_config.get(
                "landmark_model_file",
                "face_detection_yunet_2023mar_int8.onnx",
            )
        )
        landmark_path = os.path.join(models_dir, landmark_file)
        try:
            self._landmarker = cv.FaceDetectorYN.create(
                landmark_path,
                "",
                (320, 320),
                self._landmark_score_thresh,
                self._landmark_nms_thresh,
                self._landmark_top_k,
                cv.dnn.DNN_BACKEND_OPENCV,
                cv.dnn.DNN_TARGET_CPU,
            )
        except Exception:
            log.exception(
                "YuNet landmark model failed to load from %s — face embeddings disabled",
                landmark_path,
            )
            self._recognizer = None
            return

        log.info(
            "face embedder ready (model=%s, YuNet conf=%.2f, nms=%.2f)",
            self._model_name,
            self._landmark_score_thresh,
            self._landmark_nms_thresh,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def available(self) -> bool:
        return self._recognizer is not None and self._landmarker is not None

    def attach(self, frame: np.ndarray, detections: List[Detection]) -> List[Detection]:
        """Attach SFace embeddings using the landmarks already produced by YuNet."""
        if not detections:
            return detections
        if not self.available:
            return [self._with_embedding(d, None) for d in detections]

        # Do not run YuNet a second time here. The live 6 FPS detector already
        # produced the landmarks for these exact detections. Reusing them makes
        # recognition start much sooner and avoids processing a stale frame.
        order = sorted(
            range(len(detections)),
            key=lambda i: self._area(detections[i].bbox),
            reverse=True,
        )
        embeddable = set(order[: self._max_faces])

        result: List[Detection] = []
        for idx, det in enumerate(detections):
            embedding = None
            face_visible = False
            if idx in embeddable and self._is_embeddable(det):
                landmark_face = self._landmarks_from_detection(det)
                if (
                    landmark_face is not None
                    and self._landmarks_look_complete(landmark_face, det.bbox)
                    and self._face_crop_looks_visible(frame, det.bbox, landmark_face)
                ):
                    embedding = self._embed(frame, landmark_face)
                    face_visible = embedding is not None

            result.append(
                self._with_embedding(
                    det,
                    embedding,
                    face_visible=face_visible,
                )
            )

        return result

    def release(self) -> None:
        self._recognizer = None
        self._landmarker = None

    def _with_embedding(
        self,
        det: Detection,
        embedding: Optional[List[float]],
        face_visible: bool = False,
    ) -> Detection:
        return replace(
            det,
            extra={
                **det.extra,
                "embedding": embedding,
                "embedding_model": self._model_name if embedding else None,
                "face_visible": bool(face_visible),
            },
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

    @staticmethod
    def _iou(
        a: Tuple[float, float, float, float],
        b: Tuple[float, float, float, float],
    ) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        intersection = iw * ih
        if intersection <= 0.0:
            return 0.0

        union = FaceEmbedder._area(a) + FaceEmbedder._area(b) - intersection
        return intersection / union if union > 0.0 else 0.0

    @staticmethod
    def _center_distance(
        a: Tuple[float, float, float, float],
        b: Tuple[float, float, float, float],
    ) -> float:
        acx = (a[0] + a[2]) * 0.5
        acy = (a[1] + a[3]) * 0.5
        bcx = (b[0] + b[2]) * 0.5
        bcy = (b[1] + b[3]) * 0.5

        dx = acx - bcx
        dy = acy - bcy
        return float(np.hypot(dx, dy))

    def _match_yunet_face(
        self,
        faces: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        """Find the YuNet face belonging to the detector bbox.

        YuNet rows are [x, y, w, h, 10 landmarks, confidence].
        Prefer overlap; use centre distance as a tie-breaker.  A candidate
        with neither overlap nor a nearby centre is rejected to avoid using
        a neighbouring person's landmarks in crowded scenes.
        """
        best = None
        best_score = -float("inf")

        bbox_area = max(self._area(bbox), 1.0)
        bbox_diag = max(
            float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])), 1.0
        )

        for face in faces:
            if face.shape[0] < 15 or not np.isfinite(face).all():
                continue

            x, y, w, h = map(float, face[:4])
            candidate_bbox = (x, y, x + w, y + h)
            confidence = float(face[-1])
            if confidence < self._landmark_score_thresh:
                continue

            iou = self._iou(bbox, candidate_bbox)
            distance = self._center_distance(bbox, candidate_bbox)
            normalized_distance = distance / bbox_diag

            # The same face should overlap the detector bbox.  If the two
            # detectors disagree slightly, allow a nearby centre, but do not
            # cross over to a clearly separate face.
            nearby = normalized_distance <= 1.25
            if iou <= 0.01 and not nearby:
                continue

            # IoU dominates.  Confidence is only a small tie-breaker.
            score = (2.0 * iou) + (1.0 / (1.0 + normalized_distance))
            score += 0.01 * confidence

            # Avoid selecting a huge neighbouring detection merely because
            # its centre is close to a small target.
            candidate_area_ratio = self._area(candidate_bbox) / bbox_area
            if candidate_area_ratio > 4.0:
                score -= 0.25

            if score > best_score:
                best_score = score
                best = face

        return best

    @staticmethod
    def _landmarks_from_detection(det: Detection) -> Optional[np.ndarray]:
        landmarks = det.extra.get("yunet_landmarks")
        if landmarks is None:
            return None

        points = np.asarray(landmarks, dtype=np.float32).reshape(-1)
        if points.size != 10 or not np.isfinite(points).all():
            return None

        x1, y1, x2, y2 = det.bbox
        x = float(x1)
        y = float(y1)
        w = float(x2 - x1)
        h = float(y2 - y1)
        return np.concatenate(
            [
                np.asarray([x, y, w, h], dtype=np.float32),
                points,
                np.asarray([det.confidence], dtype=np.float32),
            ]
        )

    @staticmethod
    def _landmarks_look_complete(
        landmark_face: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> bool:
        """Reject heavily occluded/partial faces before recognition.

        YuNet returns right-eye, left-eye, nose and two mouth-corner
        landmarks.  A valid event requires all five points to lie inside
        the detector box and to form plausible vertical face geometry.
        This prevents detections where only the forehead/eyes remain
        visible above an occluding object such as a laptop.
        """
        if landmark_face is None or len(landmark_face) < 14:
            return False

        points = np.asarray(landmark_face[4:14], dtype=np.float32).reshape(5, 2)
        if not np.isfinite(points).all():
            return False

        x1, y1, x2, y2 = map(float, bbox)
        w = x2 - x1
        h = y2 - y1
        if w < 1.0 or h < 1.0:
            return False

        margin_x = 0.05 * w
        margin_y = 0.05 * h
        if np.any(points[:, 0] < x1 - margin_x) or np.any(points[:, 0] > x2 + margin_x):
            return False
        if np.any(points[:, 1] < y1 - margin_y) or np.any(points[:, 1] > y2 + margin_y):
            return False

        right_eye, left_eye, nose, right_mouth, left_mouth = points
        eye_y = (right_eye[1] + left_eye[1]) * 0.5
        mouth_y = (right_mouth[1] + left_mouth[1]) * 0.5

        # Eyes should be above the nose, and the mouth should be clearly
        # below the eyes.  Reject geometries that collapse into an
        # upper-face-only detection.
        if nose[1] <= eye_y:
            return False
        if mouth_y <= eye_y + (0.12 * h):
            return False
        if mouth_y <= nose[1] + (0.03 * h):
            return False

        eye_gap = float(np.linalg.norm(right_eye - left_eye))
        mouth_gap = float(np.linalg.norm(right_mouth - left_mouth))

        # YuNet can occasionally produce a high-confidence false positive
        # on hand/keyboard patterns.  Require a tighter, internally
        # consistent five-point facial layout before SFace is allowed to
        # create an embedding.
        if not 0.18 * w <= eye_gap <= 0.60 * w:
            return False
        if not 0.10 * w <= mouth_gap <= 0.45 * w:
            return False
        if not 1.25 <= eye_gap / max(mouth_gap, 1.0) <= 5.0:
            return False

        eye_mid_x = float((right_eye[0] + left_eye[0]) * 0.5)
        eye_mid_y = float((right_eye[1] + left_eye[1]) * 0.5)
        mouth_mid_x = float((right_mouth[0] + left_mouth[0]) * 0.5)
        mouth_mid_y = float((right_mouth[1] + left_mouth[1]) * 0.5)

        # A real face should have the five landmarks arranged around the
        # vertical facial axis rather than merely satisfying the ordering
        # checks above.
        if abs(eye_mid_x - ((x1 + x2) * 0.5)) > 0.12 * w:
            return False
        if abs(nose[0] - eye_mid_x) > 0.10 * w:
            return False
        if abs(mouth_mid_x - eye_mid_x) > 0.10 * w:
            return False

        eye_to_nose = (float(nose[1]) - eye_mid_y) / max(h, 1.0)
        nose_to_mouth = (mouth_mid_y - float(nose[1])) / max(h, 1.0)
        if not 0.16 <= eye_to_nose <= 0.42:
            return False
        if not 0.08 <= nose_to_mouth <= 0.26:
            return False

        face_aspect = w / max(h, 1.0)
        if not 0.65 <= face_aspect <= 1.20:
            return False

        return True

    @staticmethod
    def _face_crop_looks_visible(
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        landmark_face: np.ndarray,
    ) -> bool:
        """Reject faces whose lower region is covered by a foreground object.

        Laptop/monitor edges commonly appear as a strong horizontal boundary
        across the lower half of an otherwise valid face detection. Combine
        that boundary with the actual YuNet mouth landmarks so ordinary dark
        hair/background pixels do not cause rejection.
        """
        try:
            height, width = frame.shape[:2]
            x1, y1, x2, y2 = [int(round(value)) for value in bbox]
            x1 = max(0, min(width - 1, x1))
            x2 = max(x1 + 1, min(width, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(y1 + 1, min(height, y2))

            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] < 40 or crop.shape[1] < 40:
                return False

            gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Look for the strongest horizontal luminance boundary in the
            # middle/lower face. A monitor edge is long and coherent across
            # most of the crop width.
            best_ratio = 0.0
            best_y = -1
            for row in range(max(2, int(h * 0.35)), min(h - 2, int(h * 0.78))):
                delta = np.abs(
                    gray[row + 1, 3:-3].astype(np.int16)
                    - gray[row - 1, 3:-3].astype(np.int16)
                )
                if delta.size == 0:
                    continue
                ratio = float(np.mean(delta > 22))
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_y = row

            # Mouth landmarks are normalized by the crop bbox. If a strong
            # horizontal boundary sits above the mouth, the lower face is
            # likely hidden by a monitor/laptop.
            points = np.asarray(landmark_face[4:14], dtype=np.float32).reshape(5, 2)
            mouth_y = float((points[3, 1] + points[4, 1]) * 0.5)
            relative_mouth_y = (mouth_y - y1) / max(1.0, float(y2 - y1))

            if (
                best_ratio >= 0.58
                and 0.30 <= best_y / max(1.0, float(h)) <= 0.70
                and relative_mouth_y >= 0.52
            ):
                return False

            # Laptop keyboards/edges can produce several parallel,
            # full-width horizontal boundaries below a YuNet false-positive
            # hand detection. Reject that pattern only when the lines are
            # concentrated in the lower half of the candidate crop; this
            # leaves normal face contours/background edges untouched.
            lower_start = int(h * 0.48)
            row_scores = []
            for row in range(max(lower_start, 2), max(lower_start + 1, h - 2)):
                delta = np.abs(
                    gray[row + 1, 3:-3].astype(np.int16)
                    - gray[row - 1, 3:-3].astype(np.int16)
                )
                if delta.size:
                    row_scores.append((row, float(np.mean(delta > 18))))
            strong_rows = [
                row
                for row, ratio in row_scores
                if ratio >= 0.62
            ]
            separated = 0
            last_row = None
            for row in strong_rows:
                if last_row is None or row - last_row >= max(3, int(h * 0.025)):
                    separated += 1
                last_row = row
            if separated >= 3 and relative_mouth_y >= 0.45:
                return False

            return True
        except Exception:
            # Quality filtering must never crash face recognition.
            return True

    def _embed(
        self,
        frame: np.ndarray,
        landmark_face: np.ndarray,
    ) -> Optional[List[float]]:
        try:
            # Use the original aligned face exactly as captured.
            # No CLAHE, contrast, brightness, sharpening, or color enhancement.
            aligned = self._recognizer.alignCrop(
                frame,
                landmark_face[:-1].astype(np.float32),
            )
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
            log.exception(
                "face embedding failed for YuNet face=%s",
                landmark_face[:4].tolist() if landmark_face is not None else None,
            )
            return None
