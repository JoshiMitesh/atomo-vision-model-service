"""Person `.nb` worker — job lifecycle around the shared A311D NPU
round-robin scheduler (see npu_round_robin_scheduler.py).

The worker no longer owns its own detector or polling thread: the NPU
can only run one model instance at a time, so all `.nb` jobs share one
NpuRoundRobinScheduler/detector and this worker just registers a
FrameReader + result callback with it. Everything downstream of
detect() (face detection, upload, publish, video) still runs per-job,
on the scheduler thread, at that job's turn.
"""

from __future__ import annotations

import os

from slave.logging.logger import get_logger
from slave.models.detection import group_detections
from slave.models.frame import Frame
from slave.models.job import JobConfig
from slave.processes.detection.detection_result import DetectionResult
from slave.processes.detection.frame_annotator import draw_detections
from slave.processes.minio.frame_uploader import FrameUploader
from slave.processes.publisher.detection_publisher import DetectionPublisher
from slave.processes.stream.frame_reader import FrameReader
from slave.processes.stream.stream_connector import build_stream_url
from slave.processes.video.video_manager import VideoManager
from slave.processes.worker.face.face_controller import FaceController
from slave.processes.worker.models.person.nb import npu_round_robin_scheduler as scheduler_registry
from slave.processes.worker.worker_factory import WorkerIdentity
from slave.utils.time_utils import utc_now_iso


class PersonNbWorker:
    def __init__(
        self,
        job: JobConfig,
        model_config: dict,
        models_dir: str,
        detection_publisher: DetectionPublisher,
        frame_uploader: FrameUploader,
        video_manager: VideoManager,
        identity: WorkerIdentity,
        face_controller: FaceController,
    ) -> None:
        self._job = job
        self._detection_publisher = detection_publisher
        self._frame_uploader = frame_uploader
        self._video_manager = video_manager
        self._identity = identity
        self._face_controller = face_controller
        self._log = get_logger(__name__, job_id=job.job_id)

        library_file = model_config.get("nb", {}).get("library_file", "")
        self._model_path = os.path.join(models_dir, job.model_name)
        self._library_path = os.path.join(models_dir, library_file) if library_file else None

        url = build_stream_url(identity.stream_host, job.camera_id)
        self._frame_reader = FrameReader(job.job_id, url)

        self._scheduler = scheduler_registry.get_shared_scheduler(
            self._model_path, self._library_path, model_config, job.model_name
        )

    def start(self) -> None:
        self._frame_reader.start()
        self._scheduler.register(self._job.job_id, self._frame_reader, self._on_detections)
        self._log.info("person .nb worker started (shared NPU round-robin scheduler)")

    def stop(self) -> None:
        scheduler_registry.unregister_job(self._model_path, self._job.job_id)
        self._cleanup()
        self._log.info("person .nb worker stopped")

    def is_alive(self) -> bool:
        # No dedicated thread anymore — "alive" means still registered
        # with the shared scheduler.
        return self._job.job_id in getattr(self._scheduler, "_sources", {})

    def _on_detections(self, frame: Frame, person_detections: list) -> None:
        try:
            self._process_frame(frame, person_detections)
        except Exception:
            self._log.exception("frame processing failed for frame_id=%s", frame.frame_id)

    def _process_frame(self, frame: Frame, person_detections: list) -> None:
        face_detections = self._face_controller.update(frame.image, person_detections)
        detections = person_detections + face_detections

        # Upload happens before publishing so the frame's object_key/url
        # can be embedded directly into the DetectionResult — one merged
        # NATS message per frame instead of a separate detection message
        # plus a separate upload-started message.
        object_key = url = None
        annotated = None
        if detections:
            annotated = draw_detections(frame.image, detections)
            upload_info = self._frame_uploader.upload_frame(self._job.job_id, frame.frame_id, annotated)
            if upload_info is not None:
                object_key, url = upload_info

        # width/height mirror object_key/url — only meaningful (and
        # only sent) once there's actually an uploaded image they
        # describe; null together with object_key/url when MinIO is
        # disabled or nothing was detected this frame.
        width = height = None
        if object_key is not None:
            height, width = (int(v) for v in frame.image.shape[:2])

        # tracking_id is the frame's saved MinIO image id, not a real
        # per-object tracker — null whenever nothing was uploaded.
        telemetry = group_detections(detections, self._job, tracking_id=object_key)

        result = DetectionResult(
            job_id=self._job.job_id,
            frame_id=frame.frame_id,
            occurred_at=utc_now_iso(),
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

        # Video only records while a person is actually in frame, with
        # the same boxes already drawn above for the upload.
        if person_detections:
            self._video_manager.feed_frame(annotated)

    def _cleanup(self) -> None:
        try:
            self._frame_reader.stop()
        except Exception:
            self._log.exception("error stopping frame reader")
        # NOTE: the detector/NPU handle is NOT released here — it is
        # shared across every .nb job. scheduler_registry.unregister_job()
        # (called above, before _cleanup) already releases it, but only
        # once this was the last job using that model_path.
        try:
            self._face_controller.release()
        except Exception:
            self._log.exception("error releasing face controller")
        try:
            self._video_manager.close()
        except Exception:
            self._log.exception("error closing video manager")
