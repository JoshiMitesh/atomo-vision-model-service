"""Plate text reader: perspective-correct + CLAHE-enhance a plate crop,
then run PaddleOCR text recognition on it.

Faithful port of the reference ALPR script's find_plate_quad /
rectify_plate / enhance_plate / run_ocr pipeline (alpr_video.py handed
over for this feature) -- this is a tuned, already-working algorithm,
not something to redesign.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

import cv2
import numpy as np

from slave.logging.logger import get_logger

log = get_logger(__name__)


def _normalize_plate(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[^A-Z0-9]", "", text.upper().strip())
    return text if len(text) >= 4 else ""


def _order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    s = points.sum(axis=1)
    tl = points[np.argmin(s)]
    br = points[np.argmax(s)]
    d = np.diff(points, axis=1).reshape(-1)
    tr = points[np.argmin(d)]
    bl = points[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _quad_geometry_valid(quad: np.ndarray, crop_w: int, crop_h: int) -> bool:
    q = _order_points(quad)
    margin = 2
    if np.any(q[:, 0] <= margin) or np.any(q[:, 1] <= margin):
        return False
    if np.any(q[:, 0] >= crop_w - margin - 1) or np.any(q[:, 1] >= crop_h - margin - 1):
        return False

    width_top = np.linalg.norm(q[1] - q[0])
    width_bottom = np.linalg.norm(q[2] - q[3])
    height_left = np.linalg.norm(q[3] - q[0])
    height_right = np.linalg.norm(q[2] - q[1])
    width = (width_top + width_bottom) / 2
    height = (height_left + height_right) / 2

    if width < 15 or height < 5:
        return False
    aspect = width / height
    if aspect < 2.0 or aspect > 12.0:
        return False
    if max(width_top, width_bottom) / max(1.0, min(width_top, width_bottom)) > 2.0:
        return False
    if max(height_left, height_right) / max(1.0, min(height_left, height_right)) > 2.0:
        return False
    return True


def _find_plate_quad(crop: np.ndarray) -> Optional[np.ndarray]:
    h, w = crop.shape[:2]
    if w < 20 or h < 10:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scale = 3
    gray_big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray_big, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 180)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < w * h * 0.03 * scale * scale:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(approx) != 4:
            continue
        quad_big = approx.reshape(4, 2).astype(np.float32)
        quad = quad_big / scale
        if not _quad_geometry_valid(quad, w, h):
            continue
        _, _, rw, rh = cv2.boundingRect(quad.astype(np.float32))
        rect_area = max(1, rw * rh)
        rectangularity = area / (rect_area * scale * scale)
        score = area * max(0.1, rectangularity)
        candidates.append((score, quad))

    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0], reverse=True)
    return _order_points(candidates[0][1])


def _fallback_quad(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    x1, y1 = 0.06 * w, 0.08 * h
    x2, y2 = 0.94 * w, 0.92 * h
    return np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def _rectify(crop: np.ndarray, quad: np.ndarray, width: int, height: int) -> np.ndarray:
    destination = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(np.float32(quad), destination)
    return cv2.warpPerspective(crop, matrix, (width, height), flags=cv2.INTER_CUBIC)


def _enhance(plate: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(plate, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_chan = clahe.apply(l_chan)
    return cv2.cvtColor(cv2.merge((l_chan, a_chan, b_chan)), cv2.COLOR_LAB2BGR)


class PlateOcr:
    """Lazily imports paddleocr (mirrors UltralyticsRuntime's lazy
    `ultralytics` import) so number-plate deployments with
    `ocr.enabled: false` in config/models/number_plate.yaml don't need
    paddleocr/paddlepaddle installed at all.
    """

    def __init__(self, model_name: str, rectified_width: int, rectified_height: int) -> None:
        from paddleocr import TextRecognition

        self._recognizer = TextRecognition(model_name=model_name)
        self._rect_w = rectified_width
        self._rect_h = rectified_height

    def read(self, crop: np.ndarray) -> Tuple[str, float]:
        quad = _find_plate_quad(crop)
        if quad is None:
            quad = _fallback_quad(crop)
        rectified = _rectify(crop, quad, self._rect_w, self._rect_h)
        enhanced = _enhance(rectified)
        return self._run_ocr(enhanced)

    def _run_ocr(self, image: np.ndarray) -> Tuple[str, float]:
        try:
            results = self._recognizer.predict(image)
            for result in results:
                data = getattr(result, "json", None)
                if callable(data):
                    data = data()
                if isinstance(data, dict):
                    data = data.get("res", data)
                    text = _normalize_plate(data.get("rec_text", ""))
                    try:
                        score = float(data.get("rec_score", 0.0))
                    except (TypeError, ValueError):
                        score = 0.0
                    return text, score

                # Compatibility with other PaddleOCR versions.
                text = _normalize_plate(getattr(result, "rec_text", ""))
                try:
                    score = float(getattr(result, "rec_score", 0.0))
                except (TypeError, ValueError):
                    score = 0.0
                return text, score
        except Exception:
            log.exception("PaddleOCR inference failed")
        return "", 0.0

    def release(self) -> None:
        self._recognizer = None
