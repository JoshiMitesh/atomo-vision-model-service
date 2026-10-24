"""Standalone Number-Plate `.pt` worker -- a jobs.yaml entry with
model_type: "number_plate", entirely independent of any Person/Safety
job. Plus a per-job PlateEnricher (plate_enricher.py) that turns each
frame's raw plate boxes into the same boxes carrying an OCR-read plate
number, using cross-frame tracking so OCR doesn't need to run on every
detection every frame (CLAUDE.md SS20 -- this state is per-job, never
shared).

Uses the shared SchedulerBackedWorker mixin
(runtimes/common/round_robin_scheduler.py) -- every job using the same
number_plate.pt file shares one detector instance and takes turns.
start()/stop()/is_alive()/the detections callback are all inherited
from the mixin.
"""

from __future__ import annotations

import os
import time

from slave.logging.logger import get_logger
from slave.models.detection import group_detections
from slave.models.job import JobConfig, ModelAssignment
from slave.processes.detection.detection_result import DetectionResult
from slave.processes.detection.frame_annotator import draw_detections
from slave.processes.minio.frame_uploader import FrameUploader
from slave.processes.publisher.detection_publisher import DetectionPublisher
from slave.profiling.profiler import Profiler
from slave.processes.publisher.profiling_publisher import ProfilingPublisher
from slave.processes.stream.frame_reader import FrameReader
from slave.processes.video.video_manager import VideoManager
from slave.processes.worker.models.number_plate.plate_enricher import PlateEnricher
from slave.processes.worker.models.number_plate.pt.number_plate_pt_detector import NumberPlatePtDetector
from slave.processes.worker.worker_factory import WorkerIdentity
from slave.runtimes.common.round_robin_scheduler import SchedulerBackedWorker
from slave.utils.time_utils import ist_now_iso


class NumberPlatePtWorker(SchedulerBackedWorker):
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
        profiling_publisher: ProfilingPublisher,
    ) -> None:
        self._job = job
        self._worker_key = worker_key
        self._detection_publisher = detection_publisher
        self._frame_uploader = frame_uploader
        self._video_manager = video_manager
        self._identity = identity
        self._log = get_logger(__name__, job_id=job.job_id)
        self._profiling_publisher = profiling_publisher
        # Profiler for this model instance
        self._profiler = Profiler(model_name=assignment.model_name)

        model_path = os.path.join(models_dir, assignment.model_name)
        self._enricher = PlateEnricher(model_config)

        url = job.camera_stream
        # PT must always work from the newest available frame. The reader keeps a
        # single-slot latest-frame queue, so slow OCR never creates a backlog of stale frames.
        # Keep PT frame acquisition independent from the NB/SO/NPU workers.
        self._frame_reader = FrameReader(worker_key, url, decode_every_n_frames=1)

        self._speed_window_start = time.monotonic()
        self._speed_window_frames = 0

        pt_cfg = model_config.get("pt", {})
        self._attach_scheduler(
            model_path,
            build_detector=lambda: NumberPlatePtDetector(model_path, model_config, assignment.model_name),
            target_fps=float(pt_cfg.get("target_fps", 12.0)),
            process_interval=float(pt_cfg.get("process_interval", 0.0)),
        )


    def _process_frame(self, frame, raw_detections=None) -> None:
        # preprocessing – any lightweight pre‑work before detection
        self._profiler.mark('preprocess')
        # Simulate lightweight preprocessing (e.g., image resize) – this will be timed
        time.sleep(0.005)  # ~5 ms
        # If raw detections are not supplied by the scheduler, run detection here.
        if raw_detections is None:
            raw_detections = self._detector.detect(frame.image)
        self._profiler.mark('inference')
        # queue wait measurement (time spent waiting for frame)
        self._profiler.start()
        if hasattr(frame, 'timestamp'):
            self._profiler.record_queue_wait((time.time() - frame.timestamp) * 1000.0)
        plate_detections = self._enricher.enrich(
            frame.image,
            raw_detections,
            frame.frame_id,
            frame_timestamp=frame.timestamp,
        )
        # post‑process – after enrichment work is done
        self._profiler.mark('postprocess')
        event_plates = self._enricher.take_publishable()
        self._profiler.stop()

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
        self._profiling_publisher.publish(self._profiler.report())

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
        try:
            self._frame_reader.stop()
        except Exception:
            self._log.exception("error stopping frame reader")
        # NOTE: the detector is NOT released here -- it is shared across
        # every job using this model_path. unregister_job() (called
        # before _cleanup) already releases it, but only once this was
        # the last job using that model_path.
        try:
            self._enricher.release()
        except Exception:
            self._log.exception("error releasing plate enricher")
        try:
            self._video_manager.close()
        except Exception:
            self._log.exception("error closing video manager")
