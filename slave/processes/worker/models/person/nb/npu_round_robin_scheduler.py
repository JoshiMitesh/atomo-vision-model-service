"""Round-robin frame scheduler shared by all Person `.nb` jobs.

The A311D NPU can only run one model instance at a time (hardware
constraint) — independent per-job `NbRuntime` handles the way
`.pt`/`.so` jobs work will contend or fail once two or more Person
`.nb` jobs are active together. This scheduler holds exactly one
shared `PersonNbDetector`/NPU handle and cycles through every
registered job's `FrameReader` in turn, running one inference per
turn on that shared handle — the same pattern as the reference
`RoundRobinNPUWorker`, minus its local-JSON-file output (this project
publishes straight to NATS via each job's own DetectionPublisher
callback instead).

Everything except the NPU handle itself stays per-job (CLAUDE.md
§20): each job keeps its own FrameReader, its own DetectionResult,
its own publish/upload/video callback. Only detect() and the
underlying NPU handle are shared, and only because the hardware
forces it.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Callable, Dict, NamedTuple, Optional

from slave.logging.logger import get_logger
from slave.models.frame import Frame
from slave.processes.stream.frame_reader import FrameReader
from slave.processes.worker.models.person.nb.person_nb_detector import PersonNbDetector

log = get_logger(__name__)

# Called on the scheduler thread once per turn for a job that produced
# a frame: (frame, person_detections) -> None
OnDetections = Callable[[Frame, list], None]


class _Registration(NamedTuple):
    frame_reader: FrameReader
    on_detections: OnDetections


class NpuRoundRobinScheduler:
    """One instance per shared NPU handle. `.nb` jobs register with it
    instead of each owning their own detector + polling thread.
    """

    def __init__(self, detector: PersonNbDetector, target_fps: float = 12.0) -> None:
        self._detector = detector
        self._sources: "OrderedDict[str, _Registration]" = OrderedDict()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._index = 0
        self.total_inference_fps = 0.0
        self._log = get_logger(__name__)

        # Caps each job at target_fps rather than running the full
        # letterbox->tensor->NPU->decode pipeline on every single frame
        # the camera delivers (25-30fps native) — the CPU-heavy steps
        # (letterbox, to_input_tensor) are unconditional per detect()
        # call, so this directly cuts their total cost proportionally.
        self._min_detect_interval = 1.0 / target_fps if target_fps > 0 else 0.0
        self._last_detected: Dict[str, float] = {}

    def register(self, job_id: str, frame_reader: FrameReader, on_detections: OnDetections) -> None:
        with self._lock:
            self._sources[job_id] = _Registration(frame_reader, on_detections)
        self._ensure_started()

    def unregister(self, job_id: str) -> bool:
        """Returns True if this was the last registered job (scheduler
        thread has been stopped and the NPU handle should be released).
        """
        with self._lock:
            self._sources.pop(job_id, None)
            empty = not self._sources
        self._last_detected.pop(job_id, None)
        if empty:
            self._stop_thread()
        return empty

    def release_detector(self) -> None:
        self._detector.release()

    def _ensure_started(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._loop, name="person-nb-round-robin", daemon=True)
            self._thread.start()

    def _stop_thread(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=3.0)
        self._thread = None

    def _loop(self) -> None:
        fps_count = 0
        fps_start = time.monotonic()

        while not self._stop_event.is_set():
            with self._lock:
                job_ids = list(self._sources.keys())

            if not job_ids:
                time.sleep(0.01)
                continue

            if self._index >= len(job_ids):
                self._index = 0
            job_id = job_ids[self._index]
            self._index += 1

            with self._lock:
                reg = self._sources.get(job_id)
            if reg is None:
                continue  # unregistered mid-cycle

            # FPS cap: skip this job's turn without touching the NPU if
            # it was detected on too recently — moves straight to the
            # next job in the round-robin instead of burning a cycle,
            # so this caps each job independently at target_fps rather
            # than dividing a fixed total budget across jobs.
            now = time.monotonic()
            if now - self._last_detected.get(job_id, 0.0) < self._min_detect_interval:
                time.sleep(0.001)
                continue

            frame = reg.frame_reader.get_latest_frame(timeout=0.0)
            if frame is None:
                time.sleep(0.001)
                continue

            try:
                detections = self._detector.detect(frame.image)
            except Exception:
                self._log.exception("NPU detect failed for job_id=%s", job_id)
                time.sleep(0.01)
                continue

            self._last_detected[job_id] = now

            try:
                reg.on_detections(frame, detections)
            except Exception:
                self._log.exception("on_detections callback failed for job_id=%s", job_id)

            fps_count += 1
            now = time.monotonic()
            elapsed = now - fps_start
            if elapsed >= 2.0:
                self.total_inference_fps = fps_count / elapsed
                fps_count = 0
                fps_start = now


# One shared scheduler per distinct .nb model file — in practice this
# is a single entry (one Person model), but keyed by model_path rather
# than hardcoded to a singleton in case that ever changes.
_registry: Dict[str, NpuRoundRobinScheduler] = {}
_registry_lock = threading.Lock()


def get_shared_scheduler(
    model_path: str,
    library_path: Optional[str],
    model_config: dict,
    model_name: str,
) -> NpuRoundRobinScheduler:
    with _registry_lock:
        scheduler = _registry.get(model_path)
        if scheduler is None:
            target_fps = float(model_config.get("nb", {}).get("target_fps", 12.0))
            log.info(
                "creating shared NPU round-robin scheduler for model_path=%s (target_fps=%s)",
                model_path, target_fps,
            )
            detector = PersonNbDetector(model_path, library_path, model_config, model_name)
            scheduler = NpuRoundRobinScheduler(detector, target_fps=target_fps)
            _registry[model_path] = scheduler
        return scheduler


def unregister_job(model_path: str, job_id: str) -> None:
    """Unregister a job from its scheduler; releases the shared NPU
    handle and drops the registry entry once no job uses it anymore.
    """
    with _registry_lock:
        scheduler = _registry.get(model_path)
        if scheduler is None:
            return
        empty = scheduler.unregister(job_id)
        if empty:
            log.info("releasing shared NPU round-robin scheduler for model_path=%s", model_path)
            scheduler.release_detector()
            _registry.pop(model_path, None)
