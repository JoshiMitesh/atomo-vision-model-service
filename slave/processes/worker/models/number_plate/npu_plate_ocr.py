"""Plate crop preparation + OCR for the NPU (`.nb`/`.so`) number_plate
path.

Ported from the board-verified alpr_npu.py: contour-based corner
finding that rejects untrustworthy quads (falling back to a small inset
trapezoid), perspective warp with border replication, CLAHE + mild
sharpen, then fast-plate-ocr's CCT text recognition -- fast-plate-ocr
always runs on ONNX Runtime internally, so the board does not need
paddlepaddle (no aarch64 wheel) at all.

Independent of plate_ocr.py (the `.pt` path's OCR) in that each keeps
its own recognizer instance -- both now use the same fast-plate-ocr
backend.
"""

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
    def __init__(self, model_name: str, engine: str, out_w: int, out_h: int) -> None:
        from fast_plate_ocr import LicensePlateRecognizer  # lazy: keeps detection-only setups free of the dependency

        # `engine` is accepted for config/call-site compatibility but unused:
        # fast_plate_ocr always runs on ONNX Runtime internally. model_name
        # must be one of its hub_ocr_model literals (e.g. cct-s-v1-global-model),
        # downloaded to ~/.cache/fast-plate-ocr on first use.
        self._recognizer = LicensePlateRecognizer(hub_ocr_model=model_name)
        self._out_w = out_w
        self._out_h = out_h
        log.info("plate OCR ready (model=%s)", model_name)

    def read(self, crop: np.ndarray) -> Tuple[str, float]:
        """Crop -> perspective-corrected, enhanced plate -> (text, confidence)."""
        prepared = enhance_plate(perspective_correct(crop, self._out_w, self._out_h))
        return self._recognize(prepared)

    def _recognize(self, image: np.ndarray) -> Tuple[str, float]:
        try:
            # fast_plate_ocr's CCT models are trained on grayscale crops,
            # but the ONNX graph's input is still (H, W, 3) -- a plain
            # 1-channel array fails with "Got invalid dimensions ...
            # Expected: 3". Convert to gray for the plate-reading quality,
            # then back to 3-channel BGR for the model's expected input
            # shape (same pattern as plate_ocr.py's _run_ocr(), the
            # working .pt path).
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            result = self._recognizer.run_one(bgr, return_confidence=True)
            text = str(result.plate or "").strip()
            score = float(np.mean(result.char_probs)) if result.char_probs is not None and result.char_probs.size else 0.0
            return text, score
        except Exception:
            log.exception("plate OCR inference failed")
            return "", 0.0

    def release(self) -> None:
        self._recognizer = None
