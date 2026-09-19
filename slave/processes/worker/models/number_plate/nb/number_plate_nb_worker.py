"""Standalone Number-Plate `.nb` worker -- a jobs.yaml entry with
model_type: "number_plate", independent of any Person/Safety job.

Same structure as fire_nb_worker.py (own detector + own thread per job,
no shared scheduler) plus this model's per-job NpuPlateEnricher
(npu_plate_enricher.py: IoU tracking + OCR + evidence-weighted plate voting,
ported from the board-verified alpr_npu.py; the `.pt` worker keeps its own).

NOTE -- NPU sharing: like Fire/Face/Safety `.nb` workers (and unlike
Person's round-robin scheduler), this owns an independent NbRuntime
handle. If a Number-Plate `.nb` job runs concurrently with another
`.nb` model on the same board they drive the NPU independently; see
fire_nb_worker.py's note for the caveat and the eventual fix
(generalizing the round-robin scheduler across model types).
"""

from __future__ import annotations

import os
import threading
import time

from slave.logging.logger import get_logger
from slave.models.detection import group_detections
from slave.models.job import JobConfig, ModelAssignment
from slave.processes.detection.detection_result import DetectionResult
from slave.processes.detection.frame_annotator import draw_detections
from slave.processes.minio.frame_uploader import FrameUploader
from slave.processes.publisher.detection_publisher import DetectionPublisher
from slave.processes.stream.frame_reader import FrameReader
from slave.processes.video.video_manager import VideoManager
from slave.processes.worker.models.number_plate.nb.number_plate_nb_detector import NumberPlateNbDetector
from slave.processes.worker.models.number_plate.npu_plate_enricher import NpuPlateEnricher
from slave.processes.worker.worker_factory import WorkerIdentity
from slave.utils.time_utils import ist_now_iso

FRAME_POLL_TIMEOUT_SECONDS = 1.0


class NumberPlateNbWorker:
    def __init__(
        self,
        job: JobConfig,
        assignment: ModelAssignment,
        worker_key: str,
        model_config: dict,
        models_dir: str,
        detection_publisher: DetectionPublisher,
        frame_uploader: FrameUploader,
        video_manager: VideoManager,
        identity: WorkerIdentity,
    ) -> None:
        self._job = job
        self._worker_key = worker_key
        self._detection_publisher = detection_publisher
        self._frame_uploader = frame_uploader
        self._video_manager = video_manager
        self._identity = identity
        self._log = get_logger(__name__, job_id=job.job_id)

        library_file = model_config.get("nb", {}).get("library_file", "")
        model_path = os.path.join(models_dir, assignment.model_name)
        library_path = os.path.join(models_dir, library_file) if library_file else None
        self._detector = NumberPlateNbDetector(model_path, library_path, model_config, assignment.model_name)
        self._enricher = NpuPlateEnricher(model_config)
        target_fps = float(model_config.get("nb", {}).get("target_fps", 12))
        self._min_interval = 1.0 / target_fps if target_fps > 0 else 0.0

        url = job.camera_stream
        self._frame_reader = FrameReader(worker_key, url)

        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"number-plate-nb-{worker_key}", daemon=True
        )

    def start(self) -> None:
        self._frame_reader.start()
        self._thread.start()
        self._log.info("number_plate .nb worker started")

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=10)
        self._cleanup()
        self._log.info("number_plate .nb worker stopped")

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        # alpr_npu.py caps NPU inference at NPU_FPS (12): running the NPU
        # plus per-track OCR on every camera frame is far too heavy for
        # the board. Sleep off the rest of the interval, then take the
        # newest frame (FrameReader keeps only the latest).
        last_started = 0.0
        while not self._stop_event.is_set():
            wait = self._min_interval - (time.monotonic() - last_started)
            if wait > 0 and self._stop_event.wait(wait):
                break
            frame = self._frame_reader.get_latest_frame(timeout=FRAME_POLL_TIMEOUT_SECONDS)
            if frame is None:
                continue
            last_started = time.monotonic()
            try:
                self._process_frame(frame)
            except Exception:
                self._log.exception("frame processing failed for frame_id=%s", frame.frame_id)

    def _process_frame(self, frame) -> None:
        raw_detections = self._detector.detect(frame.image)
        plate_detections = self._enricher.enrich(frame.image, raw_detections, frame.frame_id)

        object_key = url = None
        annotated = None
        if plate_detections:
            annotated = draw_detections(frame.image, plate_detections)
            upload_info = self._frame_uploader.upload_frame(self._worker_key, frame.frame_id, annotated)
            if upload_info is not None:
                object_key, url = upload_info

        width = height = None
        if object_key is not None:
            height, width = (int(v) for v in frame.image.shape[:2])

        telemetry = group_detections(plate_detections, self._job, tracking_id=object_key)

        result = DetectionResult(
            job_id=self._job.job_id,
            frame_id=frame.frame_id,
            occurred_at=ist_now_iso(),
            branch_id=self._identity.branch_id,
            device_id=self._identity.device_id,
            zone_id=self._job.zone_id,
            camera_id=self._job.camera_id,
            camera_status=self._frame_reader.get_status(),
            telemetry=telemetry,
            object_key=object_key,
            url=url,
            width=width,
            height=height,
        )
        self._detection_publisher.publish_detection(result)

        if plate_detections:
            self._video_manager.feed_frame(annotated)

    def _cleanup(self) -> None:
        try:
            self._frame_reader.stop()
        except Exception:
            self._log.exception("error stopping frame reader")
        try:
            self._detector.release()
        except Exception:
            self._log.exception("error releasing detector")
        try:
            self._enricher.release()
        except Exception:
            self._log.exception("error releasing plate enricher")
        try:
            self._video_manager.close()
        except Exception:
            self._log.exception("error closing video manager")
