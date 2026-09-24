"""Plate text reader -- night-tuned pipeline ported from anpr_night/anpr.py.

Replaces the previous PaddleOCR perspective-correct+CLAHE pipeline with
anpr_night's proven night-CCTV approach: for each detected box, read
the plate crop under two pad ratios x two variants (raw grayscale crop,
and the same crop cleaned by plate_enhance.enhance_plate's black-hat
filter) with fast-plate-ocr, run every read through plate_format's
Indian-layout correction, then vote across the 4 reads. This is a
straight port, not a redesign -- see anpr_night/anpr.py's read_plate()/
ocr() and anpr_night/README.md for why (black-hat over CLAHE for glare,
pad variance for clipped first/last chars, vote over single-best-score).
"""

from __future__ import annotations

from collections import Counter
from typing import List, Tuple

import cv2
import numpy as np

from slave.logging.logger import get_logger
from slave.processes.worker.models.number_plate.plate_enhance import enhance_plate
from slave.processes.worker.models.number_plate.plate_format import normalize_plate

log = get_logger(__name__)

PAD_RATIOS = (0.10, 0.20)  # small pad variance fixes clipped first/last chars


class PlateOcr:
    """Lazily imports fast_plate_ocr (mirrors UltralyticsRuntime's lazy
    `ultralytics` import) so number-plate deployments with
    `ocr.enabled: false` in config/models/number_plate.yaml don't need
    fast-plate-ocr installed at all.
    """

    def __init__(self, model_name: str, pad_ratios: Tuple[float, ...] = PAD_RATIOS) -> None:
        from fast_plate_ocr import LicensePlateRecognizer

        self._recognizer = LicensePlateRecognizer(model_name)
        self._pad_ratios = tuple(pad_ratios)

    def read_plate(
        self, gray_frame: np.ndarray, bbox: Tuple[float, float, float, float]
    ) -> Tuple[str, float, bool, List[Tuple[str, bool, float, str, str, float]]]:
        """gray_frame: full-frame grayscale image (BGR2GRAY of the raw frame,
        computed once per frame by the caller). bbox: detector box in that
        frame's coordinates. Returns (best_plate, confidence, valid,
        per-read diagnostics) -- same shape as anpr_night's read_plate()."""
        h, w = gray_frame.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        reads: List[Tuple[str, bool, float, str, str, float]] = []
        for pad in self._pad_ratios:
            px, py = int(pad * (x2 - x1)), int(pad * (y2 - y1))
            crop = gray_frame[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]
            if crop.size == 0:
                continue
            for name, img in (("raw", crop), ("clean", enhance_plate(crop))):
                text, score = self._run_ocr(img)
                plate, ok = normalize_plate(text)
                reads.append((plate, ok, score, text, name, pad))

        if not reads:
            return "", 0.0, False, reads

        valid = [r for r in reads if r[1]]
        pool = valid or reads
        votes = Counter(r[0] for r in pool)
        best = max(votes, key=lambda k: (votes[k], max(r[2] for r in pool if r[0] == k)))
        conf = max(r[2] for r in pool if r[0] == best)
        return best, conf, bool(valid), reads

    def _run_ocr(self, img_gray: np.ndarray) -> Tuple[str, float]:
        try:
            result = self._recognizer.run(cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR), return_confidence=True)[0]
            score = float(np.mean(result.char_probs)) if result.char_probs is not None else 0.0
            return result.plate.replace("_", "").strip().upper(), score
        except Exception:
            log.exception("fast-plate-ocr inference failed")
            return "", 0.0

    def release(self) -> None:
        self._recognizer = None
