"""NPU plate crop preparation with CPU-only PP-OCRv6 Medium recognition."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from slave.logging.logger import get_logger

log = get_logger(__name__)


def order_points(points) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def proportional_corners(crop: np.ndarray) -> np.ndarray:
    """Conservative fallback: the YOLO box is already a plate region, so
    use a small inset trapezoid rather than risk a contour that latched
    onto the crop boundary."""
    h, w = crop.shape[:2]
    x_margin = max(1, int(w * 0.04))
    y_margin = max(1, int(h * 0.08))
    pts = np.float32(
        [
            [x_margin + w * 0.015, y_margin],
            [w - x_margin - w * 0.015, y_margin + h * 0.04],
            [w - x_margin, h - y_margin],
            [x_margin, h - y_margin - h * 0.03],
        ]
    )
    return order_points(pts)


def find_plate_corners(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    if w < 20 or h < 10:
        return proportional_corners(crop)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1.0
    crop_area = float(w * h)

    for contour in contours:
        area = cv2.contourArea(contour)
        # Avoid tiny and full-crop contours.
        if area < crop_area * 0.08 or area > crop_area * 0.92:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) != 4:
            continue

        ordered = order_points(approx.reshape(4, 2).astype(np.float32))

        avg_width = (np.linalg.norm(ordered[1] - ordered[0]) + np.linalg.norm(ordered[2] - ordered[3])) / 2.0
        avg_height = (np.linalg.norm(ordered[3] - ordered[0]) + np.linalg.norm(ordered[2] - ordered[1])) / 2.0
        if avg_height <= 0:
            continue

        aspect = avg_width / avg_height
        if aspect < 1.8 or aspect > 9.0:
            continue

        # Reject quadrilaterals touching the crop boundary.
        margin = 1.5
        if (
            np.any(ordered[:, 0] <= margin)
            or np.any(ordered[:, 1] <= margin)
            or np.any(ordered[:, 0] >= w - margin)
            or np.any(ordered[:, 1] >= h - margin)
        ):
            continue

        rectangularity = area / (avg_width * avg_height + 1e-6)
        score = rectangularity * min(area / crop_area, 1.0)
        if score > best_score:
            best_score = score
            best = ordered

    return best if best is not None else proportional_corners(crop)


def perspective_correct(crop: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    src = find_plate_corners(crop)
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        crop, matrix, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


_SHARPEN = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)


def enhance_plate(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.cvtColor(cv2.merge([clahe.apply(l_chan), a_chan, b_chan]), cv2.COLOR_LAB2BGR)
    return cv2.filter2D(enhanced, -1, _SHARPEN)


class NpuPlateOcr:
    """Keep the NPU crop-preparation interface, but use the working Medium OCR."""

    def __init__(self, model_name: str, engine: str, out_w: int, out_h: int) -> None:
        from slave.processes.worker.models.number_plate.ppocr_v6_tiny import PpOcrV6Medium

        self._recognizer = PpOcrV6Medium()
        self._out_w = out_w
        self._out_h = out_h
        log.info("NPU plate OCR ready (PP-OCRv6 Medium, ONNX Runtime, CPU)")

    def read(self, crop: np.ndarray) -> Tuple[str, float]:
        """Perspective-correct crop, then recognize using PP-OCRv6 Medium."""
        try:
            prepared = enhance_plate(perspective_correct(crop, self._out_w, self._out_h))
            h, w = prepared.shape[:2]
            text, score, valid, _ = self._recognizer.read_plate(prepared, (0, 0, w, h))
            return (text, score) if valid else ("", 0.0)
        except Exception:
            log.exception("NPU plate OCR inference failed")
            return "", 0.0

    def release(self) -> None:
        if self._recognizer is not None:
            self._recognizer.release()
            self._recognizer = None
