"""Turns a frame's raw plate Detection list into the same boxes
carrying an OCR-read plate number.

Per-job state (its own PlateTracker + fast-plate-ocr recognizer
instance, see plate_ocr.py) that the pt/nb worker calls after its own
detector's detect() --
detection/decode/NMS stay in the detector per CLAUDE.md SS8; this is
the plate-specific post-processing that needs cross-frame memory,
which a stateless detector call can't hold. One PlateEnricher per job,
never shared (CLAUDE.md SS20).
"""

from __future__ import annotations

import cv2
import time
from dataclasses import replace
from typing import List

import numpy as np

from slave.logging.logger import get_logger
from slave.models.detection import Detection
from slave.processes.worker.models.number_plate.plate_ocr import PlateOcr
from slave.processes.worker.models.number_plate.plate_tracker import PlateTracker

log = get_logger(__name__)


class PlateEnricher:
    def __init__(self, model_config: dict) -> None:
        ocr_cfg = model_config.get("ocr", {})
        tracking_cfg = model_config.get("tracking", {})

        self._interval = int(ocr_cfg.get("interval_frames", 5))
        # A plate's number is often readable in only one or two frames of
        # its pass (the rest are blurred/angled/clipped), so while a track
        # has no accepted read yet, OCR runs this often instead of
        # `interval_frames`.
        self._search_interval = int(ocr_cfg.get("search_interval_frames", 1))
        self._min_score = float(ocr_cfg.get("min_score", 0.45))
        self._min_confirmations = int(ocr_cfg.get("min_confirmations", 1))
        self._max_vote_history = int(ocr_cfg.get("max_vote_history", 30))
        self._validate_indian_plate = bool(ocr_cfg.get("validate_indian_plate", True))

        self._tracker = PlateTracker(
            max_match_distance=float(tracking_cfg.get("max_match_distance_px", 90)),
            max_missed_frames=int(tracking_cfg.get("max_missed_frames", 20)),
            max_track_gap_seconds=float(tracking_cfg.get("max_track_gap_seconds", 1.5)),
        )

        self._publish_queue: List[Detection] = []
        # Avoid repeating the same recognized plate in the Events feed when
        # tracking briefly loses a moving vehicle and creates a new track.
        self._published_plate_numbers: dict[str, float] = {}
        self._plate_event_dedupe_seconds = max(
            60.0, float(tracking_cfg.get("plate_event_dedupe_seconds", 1800))
        )
        self._ocr = None
        if bool(ocr_cfg.get("enabled", True)):
            kwargs = {}
            if ocr_cfg.get("pad_ratios"):  # [[pad_x, pad_y], ...]; omitted -> plate_ocr.PAD_RATIOS
                kwargs["pad_ratios"] = tuple(tuple(p) if isinstance(p, (list, tuple)) else p for p in ocr_cfg["pad_ratios"])
            self._ocr = PlateOcr(
                model_name=ocr_cfg.get("model_name", "cct-s-v2-global-model"),
                **kwargs,
            )

    def enrich(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        frame_number: int,
        frame_timestamp: float = 0.0,
    ) -> List[Detection]:
        pairs = self._tracker.update(
            detections,
            frame_number=frame_number,
            timestamp=float(frame_timestamp),
        )
        enriched: List[Detection] = []
        self._publish_queue = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if self._ocr is not None and pairs else None

        for detection, track in pairs:
            interval = self._interval if track.confirmation_count else self._search_interval
            if self._ocr is not None and (frame_number - track.last_ocr_frame) >= interval:
                track.last_ocr_frame = frame_number
                plate, score, valid, _ = self._ocr.read_plate(gray, detection.bbox)
                accepted = bool(plate) and score >= self._min_score
                format_ok = accepted and (not self._validate_indian_plate or valid)
                if accepted and format_ok:
                    track.record_ocr(plate, score, self._max_vote_history)
                else:
                    log.debug(
                        "track %s: OCR rejected (plate=%r score=%.3f min_score=%.2f format_ok=%s)",
                        track.track_id, plate, score, self._min_score, format_ok,
                    )

            # Don't publish a plate_number until enough accepted OCR
            # reads have been seen for this track -- a single-frame OCR
            # fluke stays null instead of leaking into the detection
            # output. current_text is a score-weighted per-character
            # consensus across every accepted read so far (see
            # plate_tracker.py's _consensus()).
            confirmed = track.current_text and track.confirmation_count >= self._min_confirmations
            enriched_detection = replace(
                detection,
                extra={
                    "plate_number": track.current_text if confirmed else None,
                    "plate_confidence": track.current_score if confirmed else None,
                },
            )
            enriched.append(enriched_detection)

            # First try OCR before creating the event. If the first OCR attempt
            # cannot read the plate, publish a detection with plate_number=null
            # so the frontend can show "No Recognition". If OCR later confirms
            # a number, publish a second event with the recognized plate.
            ocr_attempted = self._ocr is None or track.last_ocr_frame == frame_number
            if not track.detection_event_published and ocr_attempted:
                track.detection_event_published = True
                if confirmed:
                    track.published = True
                    if not self._was_plate_published(track.current_text):
                        self._publish_queue.append(enriched_detection)
                else:
                    # Keep unreadable detections visible, but recognized plates
                    # are deduplicated across newly-created tracks below.
                    self._publish_queue.append(enriched_detection)
            elif confirmed and not track.published:
                track.published = True
                if not self._was_plate_published(track.current_text):
                    self._publish_queue.append(enriched_detection)
        return enriched

    def _was_plate_published(self, plate_number: str) -> bool:
        normalized = "".join(ch for ch in plate_number.upper() if ch.isalnum())
        if not normalized:
            return False
        now = time.monotonic()
        cutoff = now - self._plate_event_dedupe_seconds
        self._published_plate_numbers = {
            plate: seen_at
            for plate, seen_at in self._published_plate_numbers.items()
            if seen_at >= cutoff
        }
        if normalized in self._published_plate_numbers:
            log.debug("suppressing duplicate number-plate event for %s", normalized)
            return True
        self._published_plate_numbers[normalized] = now
        return False

    def take_publishable(self) -> List[Detection]:
        queued = self._publish_queue
        self._publish_queue = []
        return queued

    def release(self) -> None:
        if self._ocr is not None:
            self._ocr.release()
