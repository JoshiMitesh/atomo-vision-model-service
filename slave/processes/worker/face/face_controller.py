"""FaceController — the person-triggered Face detection state machine.

Owned by a single Person worker (one instance per job_id, built by
worker_factory.py). Runs Face inference in-process on the exact frame
the Person detector already pulled off that job's stream — no second
RTSP connection, no second frame queue, no second job_id.

State machine, driven once per frame by `update()`:

    person detected             -> activate (lazy-load the Face
                                    detector on first use), run
                                    inference (subject to the
                                    configured frame-skip interval)
    person not detected,
    still within grace window   -> keep the detector warm, keep
                                    running inference (a person may be
                                    briefly occluded, not actually gone)
    person not detected,
    grace window expired        -> release() the Face detector, go idle
    idle (no person, no grace)  -> zero cost: no detector loaded, no
                                    inference call, no allocation

This mirrors CLAUDE.md §21's shutdown ordering ("release NPU/model")
for the grace-window-expiry case, and keeps per-job isolation
automatic: a fresh FaceController is constructed per worker instance,
so one job's face state/timers/detector never touch another job's
(CLAUDE.md §20).
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from slave.logging.logger import get_logger
from slave.models.detection import Detection
from slave.utils.time_utils import monotonic

DetectorFactory = Callable[[], object]


class FaceController:
    def __init__(
        self,
        job_id: str,
        detector_factory: DetectorFactory,
        off_timeout_sec: float,
        inference_frame_interval: int = 1,
    ) -> None:
        self._job_id = job_id
        self._detector_factory = detector_factory
        self._off_timeout_sec = float(off_timeout_sec)
        self._inference_frame_interval = max(1, int(inference_frame_interval))
        self._log = get_logger(__name__, job_id=job_id)

        self._detector = None
        self._active = False
        self._last_person_seen: Optional[float] = None
        self._frame_counter = 0

    def update(self, frame_image: np.ndarray, person_detections: List[Detection]) -> List[Detection]:
        """Call once per frame, after the Person detector has run.
        Returns Face detections for this frame (possibly empty).
        """
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
            return []  # idle: no person, no grace window -> zero cost

        if not self._active:
            return []  # activation failed (e.g. model load error)

        return self._maybe_infer(frame_image)

    def _maybe_infer(self, frame_image: np.ndarray) -> List[Detection]:
        self._frame_counter += 1
        if self._frame_counter % self._inference_frame_interval != 0:
            return []
        try:
            return self._detector.detect(frame_image)
        except Exception:
            self._log.exception("face detection failed for job_id=%s", self._job_id)
            return []

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
        """Release the underlying Face detector, if one is loaded. Safe
        to call even when the controller was never activated (idle for
        a job's entire lifetime) — called unconditionally from the
        owning worker's _cleanup().
        """
        if self._detector is not None:
            try:
                self._detector.release()
            except Exception:
                self._log.exception("error releasing face detector")
        self._detector = None
        self._active = False
        self._last_person_seen = None
        self._frame_counter = 0
