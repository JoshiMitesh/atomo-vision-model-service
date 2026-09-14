"""Person `.pt` worker — job lifecycle, frame loop, result delivery.

Owns everything specific to running one job: its own RTSP reader,
detector instance, stop event, and video/upload state. Never shares
state with another job's worker (CLAUDE.md §20).
"""

from __future__ import annotations

import os
import threading

from slave.logging.logger import get_logger
from slave.models.detection import group_detections
from slave.models.job import JobConfig
from slave.processes.detection.detection_result import DetectionResult
from slave.processes.detection.frame_annotator import draw_detections
from slave.processes.minio.frame_uploader import FrameUploader
from slave.processes.publisher.detection_publisher import DetectionPublisher
from slave.processes.stream.frame_reader import FrameReader
from slave.processes.stream.stream_connector import build_stream_url
from slave.processes.video.video_manager import VideoManager
from slave.processes.worker.face.face_controller import FaceController
from slave.processes.worker.models.person.pt.person_pt_detector import PersonPtDetector
from slave.processes.worker.worker_factory import WorkerIdentity
from slave.utils.time_utils import utc_now_iso

FRAME_POLL_TIMEOUT_SECONDS = 1.0


class PersonPtWorker:
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

        model_path = os.path.join(models_dir, job.model_name)
        self._detector = PersonPtDetector(model_path, model_config, job.model_name)

        url = build_stream_url(identity.stream_host, job.camera_id)
        self._frame_reader = FrameReader(job.job_id, url)

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"person-pt-{job.job_id}", daemon=True)

    def start(self) -> None:
        self._frame_reader.start()
        self._thread.start()
        self._log.info("person .pt worker started")

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=10)
        self._cleanup()
        self._log.info("person .pt worker stopped")

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            frame = self._frame_reader.get_latest_frame(timeout=FRAME_POLL_TIMEOUT_SECONDS)
            if frame is None:
                continue
            try:
                self._process_frame(frame)
            except Exception:
                self._log.exception("frame processing failed for frame_id=%s", frame.frame_id)

    def _process_frame(self, frame) -> None:
        person_detections = self._detector.detect(frame.image)
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
        try:
            self._detector.release()
        except Exception:
            self._log.exception("error releasing detector")
        try:
            self._face_controller.release()
        except Exception:
            self._log.exception("error releasing face controller")
        try:
            self._video_manager.close()
        except Exception:
            self._log.exception("error closing video manager")
