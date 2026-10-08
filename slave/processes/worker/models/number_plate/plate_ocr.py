"""PT license-plate OCR pipeline based directly on the supplied reference ALPR.

The PT path uses PaddleOCR TextRecognition, padded crops, perspective
rectification, CLAHE enhancement, Indian-format validation and returns
one best OCR read for the track to accumulate across frames.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Tuple

import cv2
import numpy as np

from slave.logging.logger import get_logger
from slave.processes.worker.models.number_plate.plate_format import normalize_plate

log = get_logger(__name__)

RECT_W = 320
RECT_H = 80
PAD_RATIOS = ((0.03, 0.03),)


class PlateOcr:
    def __init__(self, model_name: str, pad_ratios=PAD_RATIOS) -> None:
        from paddleocr import TextRecognition

        self._recognizer = TextRecognition(model_name=model_name)
        self._pad_ratios = tuple(
            (p, p) if isinstance(p, (int, float)) else tuple(p)
            for p in pad_ratios
        )

    def read_plate(
        self,
        gray_frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> Tuple[str, float, bool, List[Tuple[str, bool, float, str, str, tuple]]]:
        h, w = gray_frame.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        reads = []

        for pad_x_ratio, pad_y_ratio in self._pad_ratios:
            bw = x2 - x1
            bh = y2 - y1
            px = int(pad_x_ratio * bw)
            py = int(pad_y_ratio * bh)

            crop = gray_frame[
                max(0, y1 - py):min(h, y2 + py),
                max(0, x1 - px):min(w, x2 + px),
            ]
            if crop.size == 0:
                continue

            quad = self._find_plate_quad(crop)
            if quad is None:
                quad = self._fallback_quad(crop)

            rectified = self._rectify_plate(crop, quad)
            enhanced = self._enhance_plate(rectified)

            for name, image in (("rectified", rectified), ("enhanced", enhanced)):
                text, score = self._run_ocr(image)
                plate, ok = normalize_plate(text)
                reads.append((plate, ok, score, text, name, (pad_x_ratio, pad_y_ratio)))

        if not reads:
            return "", 0.0, False, reads

        valid = [r for r in reads if r[1]]
        pool = valid or reads
        votes = Counter(r[0] for r in pool if r[0])
        if not votes:
            return "", 0.0, False, reads

        best = max(
            votes,
            key=lambda value: (
                votes[value],
                max(r[2] for r in pool if r[0] == value),
            ),
        )
        confidence = max(r[2] for r in pool if r[0] == best)
        return best, confidence, bool(valid), reads

    def _find_plate_quad(self, crop: np.ndarray):
        h, w = crop.shape[:2]
        if w < 20 or h < 10:
            return None

        gray = crop
        scale = 3
        gray_big = cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        blur = cv2.GaussianBlur(gray_big, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 180)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < (w * h * 0.03 * scale * scale):
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            if len(approx) != 4:
                continue

            quad = (
                approx.reshape(4, 2).astype(np.float32) / scale
            )
            if not self._quad_geometry_valid(quad, w, h):
                continue
            candidates.append((area, quad))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        return self._order_points(candidates[0][1])

    def _quad_geometry_valid(self, quad, crop_w, crop_h) -> bool:
        q = self._order_points(quad)
        margin = 2

        if np.any(q[:, 0] <= margin) or np.any(q[:, 1] <= margin):
            return False
        if np.any(q[:, 0] >= crop_w - margin - 1):
            return False
        if np.any(q[:, 1] >= crop_h - margin - 1):
            return False

        width = (
            np.linalg.norm(q[1] - q[0]) +
            np.linalg.norm(q[2] - q[3])
        ) / 2
        height = (
            np.linalg.norm(q[3] - q[0]) +
            np.linalg.norm(q[2] - q[1])
        ) / 2

        if width < 20 or height < 5:
            return False

        aspect = width / height
        return 2.0 <= aspect <= 10.0

    @staticmethod
    def _fallback_quad(crop):
        h, w = crop.shape[:2]
        return np.float32([
            [0.06 * w, 0.08 * h],
            [0.94 * w, 0.08 * h],
            [0.94 * w, 0.92 * h],
            [0.06 * w, 0.92 * h],
        ])

    @staticmethod
    def _rectify_plate(crop, quad):
        destination = np.float32([
            [0, 0],
            [RECT_W - 1, 0],
            [RECT_W - 1, RECT_H - 1],
            [0, RECT_H - 1],
        ])
        matrix = cv2.getPerspectiveTransform(
            np.float32(quad), destination
        )
        return cv2.warpPerspective(
            crop, matrix, (RECT_W, RECT_H), flags=cv2.INTER_CUBIC
        )

    @staticmethod
    def _enhance_plate(plate):
        # Reference CLAHE pipeline.
        lab = cv2.cvtColor(plate, cv2.COLOR_GRAY2BGR)
        lab = cv2.cvtColor(lab, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        result = cv2.merge((l, a, b))
        return cv2.cvtColor(result, cv2.COLOR_LAB2GRAY)

    @staticmethod
    def _order_points(points):
        points = np.asarray(points, dtype=np.float32)
        s = points.sum(axis=1)
        tl = points[np.argmin(s)]
        br = points[np.argmax(s)]
        d = np.diff(points, axis=1).reshape(-1)
        tr = points[np.argmin(d)]
        bl = points[np.argmax(d)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _run_ocr(self, image: np.ndarray) -> Tuple[str, float]:
        try:
            results = self._recognizer.predict(
                cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            )
            for result in results:
                data = getattr(result, "json", None)
                if callable(data):
                    data = data()

                if isinstance(data, dict):
                    data = data.get("res", data)
                    text = data.get("rec_text", "")
                    score = data.get("rec_score", 0.0)
                else:
                    text = getattr(result, "rec_text", "")
                    score = getattr(result, "rec_score", 0.0)

                text = str(text or "").strip().upper()
                try:
                    score = float(score)
                except Exception:
                    score = 0.0
                return text, score
        except Exception:
            log.exception("PaddleOCR inference failed")
        return "", 0.0

    def release(self) -> None:
        self._recognizer = None
