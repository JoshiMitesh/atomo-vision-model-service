"""FaceController — person-triggered face detection with camera-local tracking."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np

from slave.logging.logger import get_logger
from slave.models.detection import Detection
from slave.utils.time_utils import monotonic
from slave.processes.worker.face.face_quality import passes_face_quality_gate

DetectorFactory = Callable[[], object]
BBox = Tuple[float, float, float, float]


class FaceController:
    def __init__(
        self,
        job_id: str,
        detector_factory: DetectorFactory,
        off_timeout_sec: float,
        inference_frame_interval: int = 1,
        enabled: bool = True,
    ) -> None:
        self._job_id = job_id
        self._detector_factory = detector_factory
        self._off_timeout_sec = float(off_timeout_sec)
        self._inference_frame_interval = max(1, int(inference_frame_interval))
        self._enabled = bool(enabled)
        self._log = get_logger(__name__, job_id=job_id)

        self._detector = None
        self._active = False
        self._last_person_seen: Optional[float] = None
        self._frame_counter = 0

        # Track state is deliberately job/camera local. It is not a face
        # cluster/identity and must not be used as an identity key.
        self._tracks: Dict[str, Tuple[BBox, float]] = {}
        self._track_timeout_sec = 2.0
        self._track_iou_threshold = 0.25
        self._new_track_ids: List[str] = []
        self._new_face_events: List[Tuple[Detection, str]] = []

        if not self._enabled:
            self._log.info("face detection disabled (face.trigger.enabled=false) — person-only")

    @property
    def new_track_ids(self) -> List[str]:
        return list(self._new_track_ids)

    @property
    def new_face_events(self) -> List[Tuple[Detection, str]]:
        return list(self._new_face_events)

    def update(self, frame_image: np.ndarray, person_detections: List[Detection]) -> List[Detection]:
        if not self._enabled:
            return []

        now = monotonic()

        if person_detections:
            self._last_person_seen = now
            if not self._active:
                self._activate()
        elif self._active:
            if now - (self._last_person_seen or now) >= self._off_timeout_sec:
                self._deactivate()
                return []
        else:
            return []

        if not self._active:
            return []

        detections = self._maybe_infer(frame_image)

        detections = [
            detection
            for detection in detections
            if passes_face_quality_gate(detection)
        ]
        self._new_track_ids = []
        self._new_face_events = []

        # Expire old tracks. A later appearance gets a fresh ID and therefore
        # a fresh event/photo.
        for track_id, (_, last_seen) in list(self._tracks.items()):
            if now - last_seen >= self._track_timeout_sec:
                del self._tracks[track_id]

        tracked: List[Detection] = []
        used_tracks: set[str] = set()
        self._new_face_events = []

        for detection in detections:
            best_id: Optional[str] = None
            best_iou = 0.0

            for track_id, (old_bbox, _) in self._tracks.items():
                if track_id in used_tracks:
                    continue
                score = self._bbox_iou(old_bbox, detection.bbox)
                if score > best_iou:
                    best_iou = score
                    best_id = track_id

            is_new = best_id is None or best_iou < self._track_iou_threshold
            if is_new:
                best_id = f"face_{uuid4().hex[:12]}"
                self._new_track_ids.append(best_id)

            self._tracks[best_id] = (detection.bbox, now)
            used_tracks.add(best_id)
            tracked.append(detection)
            if is_new:
                self._new_face_events.append((detection, best_id))

        return tracked

    def track_detections(self, detections: List[Detection]) -> List[Detection]:
        """Track already-detected faces without requiring person gating."""
        now = monotonic()
        self._new_track_ids = []
        self._new_face_events = []

        for track_id, (_, last_seen) in list(self._tracks.items()):
            if now - last_seen >= self._track_timeout_sec:
                del self._tracks[track_id]

        tracked: List[Detection] = []
        used_tracks: set[str] = set()

        for detection in detections:
            best_id: Optional[str] = None
            best_iou = 0.0
            for track_id, (old_bbox, _) in self._tracks.items():
                if track_id in used_tracks:
                    continue
                score = self._bbox_iou(old_bbox, detection.bbox)
                if score > best_iou:
                    best_iou = score
                    best_id = track_id

            is_new = best_id is None or best_iou < self._track_iou_threshold
            if is_new:
                best_id = f"face_{uuid4().hex[:12]}"
                self._new_track_ids.append(best_id)

            self._tracks[best_id] = (detection.bbox, now)
            used_tracks.add(best_id)
            tracked.append(detection)
            if is_new:
                self._new_face_events.append((detection, best_id))

        return tracked

    def _maybe_infer(self, frame_image: np.ndarray) -> List[Detection]:
        self._frame_counter += 1
        if self._frame_counter % self._inference_frame_interval != 0:
            return []
        try:
            return self._detector.detect(frame_image)
        except Exception:
            self._log.exception("face detection failed for job_id=%s", self._job_id)
            return []

    @staticmethod
    def _bbox_iou(a: BBox, b: BBox) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        intersection = iw * ih
        if intersection <= 0:
            return 0.0

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection
        return intersection / union if union > 0.0 else 0.0

    def _activate(self) -> None:
        try:
            self._detector = self._detector_factory()
            self._active = True
            self._frame_counter = 0
            self._log.info("face detection activated (person detected)")
        except Exception:
            self._log.exception("failed to activate face detector")
            self._detector = None
            self._active = False

    def _deactivate(self) -> None:
        self.release()
        self._log.info("face detection deactivated (no person for >=%.1fs)", self._off_timeout_sec)

    def release(self) -> None:
        if self._detector is not None:
            try:
                self._detector.release()
            except Exception:
                self._log.exception("error releasing face detector")
        self._detector = None
        self._active = False
        self._last_person_seen = None
        self._frame_counter = 0
        self._tracks.clear()
        self._new_track_ids = []
