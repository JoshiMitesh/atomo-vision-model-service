"""Standalone Number-Plate `.pt` worker -- a jobs.yaml entry with
model_type: "number_plate", entirely independent of any Person/Safety
job. Same frame-loop shape as fire_pt_worker.py, plus a per-job
PlateEnricher (plate_enricher.py) that turns each frame's raw plate
boxes into the same boxes carrying an OCR-read plate number, using
cross-frame tracking so OCR doesn't need to run on every detection
every frame (CLAUDE.md SS20 -- this state is per-job, never shared).
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
from slave.processes.worker.models.number_plate.plate_enricher import PlateEnricher
from slave.processes.worker.models.number_plate.shared_detector import acquire_shared_detector
from slave.processes.worker.models.number_plate.pt.number_plate_pt_detector import NumberPlatePtDetector
from slave.processes.worker.worker_factory import WorkerIdentity
from slave.utils.time_utils import ist_now_iso

FRAME_POLL_TIMEOUT_SECONDS = 1.0


class NumberPlatePtWorker:
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
        self._cleanup_lock = threading.Lock()
        self._cleaned = False
        self._log = get_logger(__name__, job_id=job.job_id)

        model_path = os.path.join(models_dir, assignment.model_name)
        pt_cfg = model_config.get("pt", {})
        detector_key = (
            "number_plate_pt",
            os.path.realpath(model_path),
            str(pt_cfg.get("backend", "ultralytics")),
            str(pt_cfg.get("device", "auto")),
            bool(pt_cfg.get("half", False)),
            str(pt_cfg.get("oim_model", "")),
            float(pt_cfg.get("oim_conf_thresh", 0.25)),
        )
        self._detector = acquire_shared_detector(
            detector_key,
            lambda: NumberPlatePtDetector(model_path, model_config, assignment.model_name),
        )
        self._enricher = PlateEnricher(model_config)

        url = job.camera_stream
        # PT must always work from the newest available frame. The reader keeps a
        # single-slot latest-frame queue, so slow OCR never creates a backlog of stale frames.
        # Keep PT frame acquisition independent from the NB/SO/NPU workers.
        self._frame_reader = FrameReader(worker_key, url, decode_every_n_frames=1)

        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"number-plate-pt-{worker_key}", daemon=True
        )
        self._speed_window_start = time.monotonic()
        self._speed_window_frames = 0

    def start(self) -> None:
        self._frame_reader.start()
        self._thread.start()
        self._log.info("number_plate .pt worker started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.ident is None:
            # Safely handle stop-before-start or a failed thread start.
            self._cleanup()
        else:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                self._log.error(
                    "worker thread did not stop within 10 seconds; "
                    "cleanup will run when the worker exits"
                )
            else:
                self._cleanup()
        self._log.info("number_plate .pt worker stop requested")

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                frame = self._frame_reader.get_latest_frame(timeout=FRAME_POLL_TIMEOUT_SECONDS)
                if frame is None:
                    continue
                try:
                    self._process_frame(frame)
                except Exception:
                    self._log.exception("frame processing failed for frame_id=%s", frame.frame_id)
        finally:
            # Do not release the detector while inference may still be running.
            self._cleanup()

    def _process_frame(self, frame) -> None:
        raw_detections = self._detector.detect(frame.image)
        plate_detections = self._enricher.enrich(
            frame.image,
            raw_detections,
            frame.frame_id,
            frame_timestamp=frame.timestamp,
        )
        event_plates = self._enricher.take_publishable()

        # Only plates whose number was actually read are published (and
        # uploaded) -- an unread plate would just be a null plate_number
        # event. Video still records every detected plate.
        readable_plates = event_plates

        object_key = url = None
        annotated = None
        if plate_detections:
            annotated = draw_detections(frame.image, plate_detections)
        if readable_plates:
            upload_info = self._frame_uploader.upload_frame(self._worker_key, frame.frame_id, annotated)
            if upload_info is not None:
                object_key, url = upload_info

        width = height = None
        if object_key is not None:
            height, width = (int(v) for v in frame.image.shape[:2])

        telemetry = group_detections(readable_plates, self._job, tracking_id=object_key)

        result = DetectionResult(
            job_id=self._job.job_id,
            frame_id=frame.frame_id,
            occurred_at=ist_now_iso(),
            device_name=self._identity.device_name,
            branch_id=self._job.branch_id,
            branch_name=self._job.branch_name,
            device_id=self._identity.device_id,
            zone_id=self._job.zone_id,
            zone_name=self._job.zone_name,
            camera_id=self._job.camera_id,
            camera_name=self._job.camera_name,
            camera_stream=self._job.camera_stream,
            camera_status=self._frame_reader.get_status(),
            latitude=self._job.latitude,
            longitude=self._job.longitude,
            telemetry=telemetry,
            object_key=object_key,
            url=url,
            width=width,
            height=height,
        )
        self._detection_publisher.publish_detection(result)

        # Bounding boxes are a separate, live-only stream. Do not wait for
        # OCR voting before exposing a plate that the detector already sees.
        if plate_detections:
            live_result = DetectionResult(
                job_id=self._job.job_id,
                frame_id=frame.frame_id,
                occurred_at=ist_now_iso(),
                device_name=self._identity.device_name,
                branch_id=self._job.branch_id,
                branch_name=self._job.branch_name,
                device_id=self._identity.device_id,
                zone_id=self._job.zone_id,
                zone_name=self._job.zone_name,
                camera_id=self._job.camera_id,
                camera_name=self._job.camera_name,
                camera_stream=self._job.camera_stream,
                camera_status=self._frame_reader.get_status(),
                latitude=self._job.latitude,
                longitude=self._job.longitude,
                # Always publish the true inference-frame dimensions with live
                # boxes; the frontend must scale bboxes against this exact frame.
                width=int(frame.image.shape[1]),
                height=int(frame.image.shape[0]),
                telemetry=group_detections(plate_detections, self._job, tracking_id=None),
            )
            self._detection_publisher.publish_live_detection(live_result, self._job.camera_id)

        if plate_detections:
            self._video_manager.feed_frame(annotated)

        # PT deliberately does not sleep to a fixed FPS. The reader keeps
        # only the newest frame, so inference stays real-time instead of
        # building a stale queue. Log both rates to verify live behaviour.
        self._speed_window_frames += 1
        elapsed = time.monotonic() - self._speed_window_start
        if elapsed >= 2.0:
            processing_fps = self._speed_window_frames / elapsed
            capture_fps = self._frame_reader.get_capture_fps()
            source_lag = max(0.0, time.time() - float(frame.timestamp))
            self._log.info(
                "PT speed: capture_fps=%.1f processing_fps=%.1f source_lag=%.3fs latest_frame=true",
                capture_fps,
                processing_fps,
                source_lag,
            )
            self._speed_window_frames = 0
            self._speed_window_start = time.monotonic()

    def _cleanup(self) -> None:
        # stop() and _run()'s finally block may both request cleanup.
        with self._cleanup_lock:
            if self._cleaned:
                return
            self._cleaned = True
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
