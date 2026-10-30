"""Run OCR on the first detected plate crop and publish recognized reads immediately."""

from __future__ import annotations

import cv2
import queue
import threading
from dataclasses import replace
from typing import List

import numpy as np

from slave.logging.logger import get_logger
from slave.models.detection import Detection
from slave.processes.worker.models.number_plate.plate_ocr import PlateOcr
from slave.processes.worker.models.number_plate.awiros_ocr import AwirosPlateOcr
from slave.processes.worker.models.number_plate.plate_tracker import PlateTracker

log = get_logger(__name__)


class PlateEnricher:
    def __init__(self, model_config: dict) -> None:
        ocr_cfg = model_config.get("ocr", {})
        tracking_cfg = model_config.get("tracking", {})

        self._interval = int(ocr_cfg.get("interval_frames", 5))
        self._search_interval = int(ocr_cfg.get("search_interval_frames", 1))
        self._min_score = float(ocr_cfg.get("min_score", 0.45))
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

        self._publish_queue: List[Detection] = []
        self._publish_lock = threading.Lock()
        self._ocr_queue: queue.Queue = queue.Queue(maxsize=2)
        self._ocr_stop = threading.Event()
        self._ocr_thread = None
        self._ocr = None
        if bool(ocr_cfg.get("enabled", True)):
            kwargs = {}
            if ocr_cfg.get("pad_ratios"):
                kwargs["pad_ratios"] = tuple(
                    tuple(p) if isinstance(p, (list, tuple)) else p
                    for p in ocr_cfg["pad_ratios"]
                )
            self._ocr = (AwirosPlateOcr(
                model_dir=ocr_cfg.get("model_dir", "slave/processes/worker/models/number_plate/awiros"),
                device=ocr_cfg.get("device", "cpu"),
            ) if ocr_cfg.get("engine") == "awiros_paddle" else PlateOcr(
                model_name=ocr_cfg.get("model_name", "cct-s-v2-global-model"), **kwargs,
            ))
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
        gray = (frame if isinstance(self._ocr, AwirosPlateOcr) else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) if self._ocr is not None and pairs else None

        for detection, track in pairs:
            interval = self._interval if track.confirmation_count else self._search_interval
            if self._ocr is not None and (frame_number - track.last_ocr_frame) >= interval:
                track.last_ocr_frame = frame_number
                # OCR is isolated from the detector loop. Keep only the newest
                # pending crops so CPU-heavy recognition cannot build a backlog.
                item = (gray.copy(), detection.bbox, track, detection)
                try:
                    self._ocr_queue.put_nowait(item)
                except queue.Full:
                    try:
                        self._ocr_queue.get_nowait()
                        self._ocr_queue.task_done()
                    except queue.Empty:
                        pass
                    try:
                        self._ocr_queue.put_nowait(item)
                    except queue.Full:
                        pass

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
                frame, bbox, track, detection = self._ocr_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                plate, score, valid, _ = self._ocr.read_plate(frame, bbox)
                if plate:
                    track.record_candidate(plate, score)
                accepted = bool(plate) and score > 0.90
                format_ok = accepted and (not self._validate_indian_plate or valid)
                if not format_ok:
                    log.debug(
                        "Plate OCR not published: plate=%r confidence=%.1f%% required=>90%% format_ok=%s",
                        plate, score * 100.0, format_ok,
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
                    self._publish_queue.append(replace(detection, extra=extra))
            except Exception:
                log.exception("Background plate OCR failed for track_id=%s", track.track_id)
            finally:
                self._ocr_queue.task_done()

    def take_publishable(self) -> List[Detection]:
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
