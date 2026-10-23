"""Generic single-worker round-robin scheduler, shared by every model
type and every extension (`.nb` AND `.pt`/`.so`) that wants it --
instead of each model carrying its own near-identical copy.

Two different reasons a set of jobs might need this, both solved the
same way (one shared detector instance, jobs take turns):

1. HARDWARE HANDLE LIMIT (why this started, for `.nb`): the A311D NPU
   can only run one model instance at a time -- independent per-job
   `NbRuntime` handles the way `.pt`/`.so` jobs work will
   contend/deadlock once two or more `.nb` jobs using the SAME model
   file are active together. Confirmed live: two concurrent
   number_plate.nb jobs each got exactly one successful NPU inference
   at startup, then both went silent -- no further inference, no
   detections, no errors -- for the rest of the run.

2. RESOURCE LOAD ON WEAKER HARDWARE (why `.pt` wants this too): `.pt`
   has no hardware handle limit -- multiple independent `.pt` workers
   already run fine in parallel on capable hardware. But on a
   lower-power board, N independent full-speed `.pt` threads (each
   with its own loaded model) can overload the CPU and the whole
   process stalls. Sharing one `.pt` detector across jobs the same way
   caps total concurrent inference work to one job's turn at a time,
   same effect as the NPU case even though nothing forces it.

This scheduler holds exactly one shared detector instance per model
file and cycles through every registered job's `FrameReader` in turn,
running one `detect()` per turn on that shared instance. It is generic
over the detector: anything with a `detect(frame_image) -> List` and a
`release()` method works -- the scheduler never constructs the
detector itself, the caller passes a zero-arg factory, so this module
has no per-model-type or per-extension knowledge at all.

Everything EXCEPT the shared detector itself stays per-job (CLAUDE.md
SS20): each job keeps its own FrameReader, its own tracking/OCR/
enrichment state (where applicable), its own DetectionResult, its own
publish/upload/video callback. Only `detect()` and the underlying
detector/handle are shared -- the callback registered here receives
each frame's raw (pre-enrichment) detections; per-job enrichment (e.g.
Number-Plate's OCR+tracking) happens in the caller's own
on_detections callback, same as every other per-job step.

NOTE -- this registry is keyed by model_path, not model_type or
extension: two jobs sharing the exact same model file (e.g. two
Number-Plate `.nb` jobs on different cameras, or two Number-Plate
`.pt` jobs) correctly share one instance. Two DIFFERENT model files
(e.g. Person.nb + Number-Plate.nb running together, or two different
`.pt` files) still each get their own independent instance and, for
the `.nb` hardware-handle case specifically, can still contend with
each other for the physical NPU -- generalizing to a single instance
shared across every model file too is the further fix if that
combination needs to be supported.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Callable, Dict, NamedTuple, Optional, Protocol

from slave.logging.logger import get_logger
from slave.models.frame import Frame
from slave.processes.stream.frame_reader import FrameReader

log = get_logger(__name__)

# Called on the scheduler thread once per turn for a job that produced
# a frame: (frame, raw_detections) -> None
OnDetections = Callable[[Frame, list], None]


class SharedDetector(Protocol):
    def detect(self, frame_image) -> list: ...
    def release(self) -> None: ...


class _Registration(NamedTuple):
    frame_reader: FrameReader
    on_detections: OnDetections


class RoundRobinScheduler:
    """One instance per shared detector. Jobs register with it instead
    of each owning their own detector + polling thread.
    """

    def __init__(
        self,
        detector: SharedDetector,
        target_fps: float = 12.0,
        process_interval: float = 0.0,
    ) -> None:
        self._detector = detector
        self._sources: "OrderedDict[str, _Registration]" = OrderedDict()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._index = 0
        self._turn_count = 0
        self._last_status_log = 0.0
        self.total_inference_fps = 0.0
        self._log = get_logger(__name__)

        # Caps each job at target_fps rather than running the full
        # detect() pipeline on every single frame the camera delivers.
        fps_interval = 1.0 / target_fps if target_fps > 0 else 0.0
        requested_interval = float(process_interval) if process_interval > 0 else 0.0
        self._min_detect_interval = max(fps_interval, requested_interval)
        self._last_detected: Dict[str, float] = {}

    def register(self, job_id: str, frame_reader: FrameReader, on_detections: OnDetections) -> None:
        with self._lock:
            self._sources[job_id] = _Registration(frame_reader, on_detections)
        self._ensure_started()

    def unregister(self, job_id: str) -> bool:
        """Returns True if this was the last registered job (scheduler
        thread has been stopped and the detector should be released).
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
            self._thread = threading.Thread(target=self._loop, name="round-robin-scheduler", daemon=True)
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
            self._turn_count += 1

            with self._lock:
                reg = self._sources.get(job_id)
            if reg is None:
                continue  # unregistered mid-cycle

            # FPS cap: skip this job's turn without touching the
            # detector if it was detected on too recently -- moves
            # straight to the next job in the round-robin instead of
            # burning a cycle, so this caps each job independently at
            # target_fps rather than dividing a fixed total budget
            # across jobs.
            now = time.monotonic()
            elapsed_since_last = now - self._last_detected.get(job_id, 0.0)
            if elapsed_since_last < self._min_detect_interval:
                time.sleep(0.001)
                continue

            frame = reg.frame_reader.get_latest_frame(timeout=0.02)
            if frame is None:
                time.sleep(0.01)
                continue

            try:
                detections = self._detector.detect(frame.image)
            except Exception:
                self._log.exception("detect failed for job_id=%s", job_id)
                time.sleep(0.01)
                continue

            self._last_detected[job_id] = now
            if now - self._last_status_log >= 10.0:
                self._last_status_log = now
                self._log.info(
                    "detection active job_id=%s frame_id=%s detections=%d inference_fps=%.2f",
                    job_id,
                    frame.frame_id,
                    len(detections),
                    self.total_inference_fps,
                )

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


# One shared scheduler per distinct model file, regardless of which
# model TYPE or extension registers it -- any two jobs that happen to
# pass the same model_path share one instance. In practice this is
# per-model-type-and-extension in effect, since different
# types/extensions use different files, but nothing here is
# type-specific.
_registry: Dict[str, RoundRobinScheduler] = {}
_registry_lock = threading.Lock()


def get_shared_scheduler(
    model_path: str,
    build_detector: Callable[[], SharedDetector],
    target_fps: float = 12.0,
    process_interval: float = 0.0,
) -> RoundRobinScheduler:
    """Get (or lazily create) the shared scheduler for `model_path`.

    `build_detector` is called at most once, only the first time this
    model_path is registered -- every later call for the same
    model_path reuses the existing scheduler/detector untouched.
    """
    with _registry_lock:
        scheduler = _registry.get(model_path)
        if scheduler is None:
            log.info(
                "creating shared round-robin scheduler for model_path=%s (target_fps=%s, process_interval=%s)",
                model_path, target_fps, process_interval,
            )
            detector = build_detector()
            scheduler = RoundRobinScheduler(
                detector,
                target_fps=target_fps,
                process_interval=process_interval,
            )
            _registry[model_path] = scheduler
        return scheduler


def unregister_job(model_path: str, job_id: str) -> None:
    """Unregister a job from its scheduler; releases the shared
    detector and drops the registry entry once no job uses it anymore.
    """
    with _registry_lock:
        scheduler = _registry.get(model_path)
        if scheduler is None:
            return
        empty = scheduler.unregister(job_id)
        if empty:
            log.info("releasing shared round-robin scheduler for model_path=%s", model_path)
            scheduler.release_detector()
            _registry.pop(model_path, None)


class SchedulerBackedWorker:
    """Mixin providing the register/unregister/is_alive/callback wiring
    every scheduler-backed worker needs -- so that wiring is written
    ONCE here, not copy-pasted into every model's `_nb_worker.py`/
    `_pt_worker.py`. A worker class using this mixin only has to:

    1. In its own __init__, after setting self._log/self._worker_key/
       self._frame_reader, call self._attach_scheduler(model_path,
       build_detector, target_fps, process_interval) instead of
       constructing its own detector/thread.
    2. Implement _process_frame(self, frame, raw_detections) with its
       own model-specific detection handling (exactly what it already
       did in its old, non-shared _process_frame -- this mixin never
       touches that logic).
    3. Implement _cleanup(self) same as before, just without
       `self._detector.release()` -- the shared detector is released
       by the scheduler registry itself, only once the last job using
       that model_path has unregistered.

    start()/stop()/is_alive()/the on_detections callback are then
    inherited, identical for every worker that uses this mixin.
    """

    def _attach_scheduler(
        self,
        model_path: str,
        build_detector: "Callable[[], SharedDetector]",
        target_fps: float,
        process_interval: float = 0.0,
    ) -> None:
        self._model_path = model_path
        self._scheduler = get_shared_scheduler(
            model_path, build_detector=build_detector, target_fps=target_fps, process_interval=process_interval
        )

    def start(self) -> None:
        self._frame_reader.start()
        self._scheduler.register(self._worker_key, self._frame_reader, self._on_scheduler_detections)
        self._log.info("worker started (shared round-robin scheduler)")

    def stop(self) -> None:
        unregister_job(self._model_path, self._worker_key)
        self._cleanup()
        self._log.info("worker stopped")

    def is_alive(self) -> bool:
        return self._worker_key in getattr(self._scheduler, "_sources", {})

    def _on_scheduler_detections(self, frame: Frame, raw_detections: list) -> None:
        try:
            self._process_frame(frame, raw_detections)
        except Exception:
            self._log.exception("frame processing failed for frame_id=%s", frame.frame_id)
