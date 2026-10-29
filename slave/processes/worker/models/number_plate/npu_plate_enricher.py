"""Per-job tracking + OCR + plate-number voting for the NPU
(`.nb`/`.so`) number_plate workers -- the counterpart of
plate_enricher.py (which the `.pt` worker keeps using, untouched).

Follows the board-verified alpr_npu.py main loop: for every active
track each processed frame -> crop -> perspective-correct + enhance ->
OCR -> add the read to the track -> take the track's best result. The
detector's boxes come from the NPU (npu_decoder.py); this class only
post-processes them.

Same interface as PlateEnricher.enrich() so the workers are drop-in.
Like the reference, a failed OCR initialisation does not stop the job:
it logs and continues as detection-only (plate_number stays null).
One instance per job, never shared (CLAUDE.md SS20).
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Tuple

import numpy as np

from slave.logging.logger import get_logger
from slave.models.detection import Detection
from slave.processes.worker.models.number_plate.npu_plate_ocr import NpuPlateOcr
from slave.processes.worker.models.number_plate.npu_plate_text import is_indian_plate
from slave.processes.worker.models.number_plate.npu_plate_tracker import NpuPlateTracker

log = get_logger(__name__)


class NpuPlateEnricher:
    def __init__(self, model_config: dict) -> None:
        cfg = model_config.get("npu_ocr", {})
        shared_ocr = model_config.get("ocr", {})

        self._min_ocr_conf = float(cfg.get("min_ocr_conf", 0.45))
        self._publish_only_stable = bool(cfg.get("publish_only_stable", True))
        self._ocr_interval_frames = max(1, int(cfg.get("interval_frames", 3)))
        self._publish_queue: List[Detection] = []

        self._tracker = NpuPlateTracker(
            iou_threshold=float(cfg.get("track_iou_threshold", 0.20)),
            max_missed=int(cfg.get("track_max_missed", 18)),
            max_history=int(cfg.get("max_history", 25)),
            min_valid_observations=int(cfg.get("min_valid_observations", 3)),
            min_stable_score=float(cfg.get("min_stable_score", 1.8)),
        )

        self._ocr: Optional[NpuPlateOcr] = None
        if bool(cfg.get("enabled", True)):
            try:
                self._ocr = NpuPlateOcr(
                    model_name=str(cfg.get("model_name", "PP-OCRv5_mobile_rec")),
                    engine=str(cfg.get("engine", "onnxruntime")),
                    out_w=int(shared_ocr.get("rectified_width_px", 320)),
                    out_h=int(shared_ocr.get("rectified_height_px", 80)),
                )
            except Exception:
                log.exception("plate OCR failed to initialise -- continuing detection-only (plate_number will be null)")

    def enrich(self, frame: np.ndarray, detections: List[Detection], frame_number: int) -> List[Detection]:
        enriched: List[Detection] = []

        for detection, track in self._tracker.update(detections):
            if self._ocr is not None and frame_number - track.last_ocr_frame >= self._ocr_interval_frames:
                track.last_ocr_frame = frame_number
                raw_text, raw_conf = self._read_plate(frame, detection.bbox)
                if raw_text and raw_conf >= self._min_ocr_conf:
                    track.add_ocr(raw_text, raw_conf)

            best_text, best_conf, stable, _ = track.get_best_result()
            publish = bool(best_text) and (stable if self._publish_only_stable else is_indian_plate(best_text))

            enriched_detection = replace(
                replace(
                    detection,
                    extra={
                        "plate_number": best_text if publish else None,
                        "plate_confidence": best_conf if publish else None,
                    },
                )
            )
            enriched.append(enriched_detection)
            if publish and not track.published:
                track.published = True
                self._publish_queue.append(enriched_detection)
        return enriched

    def take_publishable(self) -> List[Detection]:
        queued = self._publish_queue
        self._publish_queue = []
        return queued

    def _read_plate(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Tuple[str, float]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
        x2, y2 = max(0, min(w, x2)), max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return "", 0.0
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "", 0.0
        try:
            return self._ocr.read(crop)
        except Exception:
            log.exception("plate crop preparation failed")
            return "", 0.0

    def release(self) -> None:
        if self._ocr is not None:
            self._ocr.release()
