"""Run OCR on the first detected plate crop and publish recognized reads immediately."""

from __future__ import annotations

import cv2
import queue
import threading
import time
from dataclasses import replace
from typing import List

import numpy as np

from slave.logging.logger import get_logger
from slave.models.detection import Detection
from slave.processes.worker.models.number_plate.ppocr_v6_tiny import PpOcrV6Medium
from slave.processes.worker.models.number_plate.plate_tracker import PlateTracker

log = get_logger(__name__)


class PlateEnricher:
    def __init__(self, model_config: dict) -> None:
        ocr_cfg = model_config.get("ocr", {})
        tracking_cfg = model_config.get("tracking", {})

        self._interval = int(ocr_cfg.get("interval_frames", 5))
        self._search_interval = int(ocr_cfg.get("search_interval_frames", 1))
        self._min_score = float(ocr_cfg.get("min_score", 0.45))
        self._publish_confidence = float(ocr_cfg.get("publish_confidence", 0.95))
        self._min_confirmations = int(ocr_cfg.get("min_confirmations", 1))
        self._min_consensus_ratio = float(ocr_cfg.get("min_consensus_ratio", 0.60))
        self._min_hits = int(ocr_cfg.get("min_detection_hits", 3))
        self._max_vote_history = int(ocr_cfg.get("max_vote_history", 30))
        self._validate_indian_plate = bool(ocr_cfg.get("validate_indian_plate", True))
        self._unrecognized_after_frames = max(1, int(ocr_cfg.get("unrecognized_after_frames", 3)))

        self._tracker = PlateTracker(
            max_match_distance=float(tracking_cfg.get("max_match_distance_px", 90)),
            max_missed_frames=int(tracking_cfg.get("max_missed_frames", 20)),
            max_track_gap_seconds=float(tracking_cfg.get("max_track_gap_seconds", 1.5)),
        )

        self._publish_queue: List[tuple[Detection, np.ndarray, int]] = []
        self._publish_lock = threading.Lock()
        self._ocr_queue: queue.Queue = queue.Queue(maxsize=16)
        # A track can have at most one OCR request queued or running. This
        # prevents slow CPU inference from filling the queue with stale crops.
        self._pending_track_ids: set[int] = set()
        self._pending_lock = threading.Lock()
        self._ocr_stop = threading.Event()
        self._ocr_thread = None
        self._ocr = None
        if bool(ocr_cfg.get("enabled", True)):
            self._ocr = PpOcrV6Medium(cpu_threads=int(ocr_cfg.get("cpu_threads", 2)))
            self._ocr_thread = threading.Thread(target=self._ocr_loop, name="plate-ocr", daemon=True)
            self._ocr_thread.start()

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
        gray = frame if self._ocr is not None and pairs else None

        for detection, track in pairs:
            interval = self._interval if track.confirmation_count else self._search_interval
            if self._ocr is not None and (frame_number - track.last_ocr_frame) >= interval:
                # Do not enqueue repeated crops while this track is already in
                # flight. A fresh frame can be queued after this request finishes.
                with self._pending_lock:
                    already_pending = track.track_id in self._pending_track_ids
                    if not already_pending:
                        self._pending_track_ids.add(track.track_id)
                if not already_pending:
                    queued_at = time.perf_counter()
                    item = (gray.copy(), detection.bbox, track, detection, frame.copy(), frame_number, queued_at)
                    try:
                        self._ocr_queue.put_nowait(item)
                        track.last_ocr_frame = frame_number
                        log.debug("Plate OCR queued: track_id=%s frame=%s queue_depth=%s",
                                  track.track_id, frame_number, self._ocr_queue.qsize())
                    except queue.Full:
                        with self._pending_lock:
                            self._pending_track_ids.discard(track.track_id)
                        # Do not mark this frame as processed; the next frame can retry.
                        log.warning("Plate OCR queue full; retrying track_id=%s", track.track_id)

            confirmed = bool(
                track.current_text
                and track.confirmation_count >= self._min_confirmations
                and track.confirmation_count / max(1, len(track.history)) >= self._min_consensus_ratio
            )
            visible_plate = track.current_text if confirmed else (track.candidate_text or track.current_text)
            visible_score = track.current_score if confirmed else (
                track.candidate_score if track.candidate_text else track.current_score
            )
            unrecognized = not visible_plate and track.detection_observations >= self._unrecognized_after_frames
            enriched.append(replace(
                detection,
                extra={
                    "plate_track_id": track.track_id,
                    "plate_number": visible_plate or None,
                    "plate_confidence": visible_score if visible_plate else None,
                    "plate_confirmed": bool(confirmed),
                    "plate_status": "confirmed" if confirmed else "unrecognized" if unrecognized else "not_confirmed",
                },
            ))
        return enriched

    def _ocr_loop(self) -> None:
        while not self._ocr_stop.is_set():
            try:
                frame, bbox, track, detection, source_image, source_frame_number, queued_at = self._ocr_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                started_at = time.perf_counter()
                queue_wait_ms = (started_at - queued_at) * 1000.0
                plate, score, valid, _ = self._ocr.read_plate(frame, bbox)
                inference_ms = (time.perf_counter() - started_at) * 1000.0
                log.info(
                    "Plate OCR result: track_id=%s frame=%s plate=%r confidence=%.4f valid=%s",
                    track.track_id, source_frame_number, plate, score, valid,
                )
                log.info(
                    "Plate OCR timing: track_id=%s source_frame=%s queue_wait_ms=%.1f inference_ms=%.1f total_ms=%.1f",
                    track.track_id, source_frame_number, queue_wait_ms, inference_ms,
                    (time.perf_counter() - queued_at) * 1000.0,
                )
                if plate:
                    track.record_candidate(plate, score)
                accepted = bool(plate) and score > self._publish_confidence
                format_ok = accepted and (not self._validate_indian_plate or valid)
                if not format_ok:
                    log.debug(
                        "Plate OCR not published: plate=%r confidence=%.1f%% required=>%.1f%% format_ok=%s",
                        plate, score * 100.0, self._publish_confidence * 100.0, format_ok,
                    )
                    continue
                track.record_ocr(plate, score, self._max_vote_history)
                log.info("Plate recognized: plate=%s confidence=%.1f%% track_id=%s",
                         plate, score * 100.0, track.track_id)
                if track.detection_event_published and track.last_event_plate == plate:
                    continue
                track.detection_event_published = True
                track.last_event_plate = plate
                extra = dict(detection.extra or {})
                extra.update({
                    "plate_track_id": track.track_id,
                    "plate_number": plate.strip().upper(),
                    "plate_confidence": float(score),
                    "plate_confirmed": True,
                    "plate_status": "confirmed",
                })
                with self._publish_lock:
                    self._publish_queue.append((replace(detection, extra=extra), source_image, source_frame_number))
            except Exception:
                log.exception("Background plate OCR failed for track_id=%s", track.track_id)
            finally:
                with self._pending_lock:
                    self._pending_track_ids.discard(track.track_id)
                self._ocr_queue.task_done()

    def take_publishable(self) -> List[tuple[Detection, np.ndarray, int]]:
        with self._publish_lock:
            queued = self._publish_queue
            self._publish_queue = []
        return queued

    def release(self) -> None:
        self._ocr_stop.set()
        if self._ocr_thread is not None:
            self._ocr_thread.join(timeout=2.0)
        if self._ocr is not None:
            self._ocr.release()
