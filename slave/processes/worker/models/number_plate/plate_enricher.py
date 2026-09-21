"""Turns a frame's raw plate Detection list into the same boxes
carrying an OCR-read plate number.

Per-job state (its own PlateTracker + PaddleOCR recognizer instance)
that the pt/nb worker calls after its own detector's detect() --
detection/decode/NMS stay in the detector per CLAUDE.md SS8; this is
the plate-specific post-processing that needs cross-frame memory,
which a stateless detector call can't hold. One PlateEnricher per job,
never shared (CLAUDE.md SS20).
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Tuple

import numpy as np

from slave.logging.logger import get_logger
from slave.models.detection import Detection
from slave.processes.worker.models.number_plate.plate_format import is_valid_indian_plate
from slave.processes.worker.models.number_plate.plate_ocr import PlateOcr
from slave.processes.worker.models.number_plate.plate_tracker import PlateTracker

log = get_logger(__name__)


class PlateEnricher:
    def __init__(self, model_config: dict) -> None:
        ocr_cfg = model_config.get("ocr", {})
        tracking_cfg = model_config.get("tracking", {})

        self._interval = int(ocr_cfg.get("interval_frames", 5))
        self._min_score = float(ocr_cfg.get("min_score", 0.45))
        self._min_confirmations = int(ocr_cfg.get("min_confirmations", 1))
        self._max_vote_history = int(ocr_cfg.get("max_vote_history", 30))
        self._validate_indian_plate = bool(ocr_cfg.get("validate_indian_plate", True))

        self._tracker = PlateTracker(
            max_match_distance=float(tracking_cfg.get("max_match_distance_px", 90)),
            max_missed_frames=int(tracking_cfg.get("max_missed_frames", 20)),
        )

        self._ocr = None
        if bool(ocr_cfg.get("enabled", True)):
            self._ocr = PlateOcr(
                model_name=ocr_cfg.get("model_name", "PP-OCRv5_mobile_rec"),
                rectified_width=int(ocr_cfg.get("rectified_width_px", 320)),
                rectified_height=int(ocr_cfg.get("rectified_height_px", 80)),
            )

    def enrich(self, frame: np.ndarray, detections: List[Detection], frame_number: int) -> List[Detection]:
        pairs = self._tracker.update(detections)
        enriched: List[Detection] = []

        for detection, track in pairs:
            if self._ocr is not None and (frame_number - track.last_ocr_frame) >= self._interval:
                track.last_ocr_frame = frame_number
                text, score = self._read_plate(frame, detection.bbox)
                accepted = bool(text) and score >= self._min_score
                format_ok = accepted and (not self._validate_indian_plate or is_valid_indian_plate(text))
                if accepted and format_ok:
                    track.record_ocr(text, score, self._max_vote_history)
                else:
                    log.debug(
                        "track %s: OCR rejected (text=%r score=%.3f min_score=%.2f format_ok=%s)",
                        track.track_id, text, score, self._min_score, format_ok,
                    )

            # Don't publish a plate_number until enough accepted OCR
            # reads have been seen for this track -- a single-frame OCR
            # fluke stays null instead of leaking into the detection
            # output. current_text is a score-weighted per-character
            # consensus across every accepted read so far (see
            # plate_tracker.py's _consensus()).
            confirmed = track.current_text and track.confirmation_count >= self._min_confirmations
            enriched.append(
                replace(
                    detection,
                    extra={
                        "plate_number": track.current_text if confirmed else None,
                        "plate_confidence": track.current_score if confirmed else None,
                    },
                )
            )
        return enriched

    def _read_plate(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Tuple[str, float]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "", 0.0
        return self._ocr.read(crop)

    def release(self) -> None:
        if self._ocr is not None:
            self._ocr.release()
