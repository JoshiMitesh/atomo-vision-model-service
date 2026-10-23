"""Person `.pt` worker — job lifecycle, frame loop, result delivery.

Uses the shared SchedulerBackedWorker mixin
(runtimes/common/round_robin_scheduler.py) -- every job using the same
person.pt file shares one detector instance and takes turns. This
caps total concurrent `.pt` inference work on weaker boards, where N
independent full-speed `.pt` threads can overload the CPU.
start()/stop()/is_alive()/the detections callback are all inherited
from the mixin.
"""

from __future__ import annotations

import os

from slave.logging.logger import get_logger
from slave.models.detection import group_detections
from slave.models.job import JobConfig, ModelAssignment
from slave.processes.detection.detection_result import DetectionResult
from slave.processes.detection.frame_annotator import draw_detections
from slave.processes.minio.frame_uploader import FrameUploader
from slave.processes.publisher.detection_publisher import DetectionPublisher
from slave.processes.stream.frame_reader import FrameReader
from slave.processes.video.video_manager import VideoManager
from slave.processes.worker.face.face_controller import FaceController
from slave.processes.worker.models.person.pt.person_pt_detector import PersonPtDetector
from slave.processes.worker.worker_factory import WorkerIdentity
from slave.runtimes.common.round_robin_scheduler import SchedulerBackedWorker
from slave.utils.time_utils import ist_now_iso


class PersonPtWorker(SchedulerBackedWorker):
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
        face_controller: FaceController,
    ) -> None:
        self._job = job
        self._worker_key = worker_key
        self._detection_publisher = detection_publisher
        self._frame_uploader = frame_uploader
        self._video_manager = video_manager
        self._identity = identity
        self._face_controller = face_controller
        self._log = get_logger(__name__, job_id=job.job_id)

        model_path = os.path.join(models_dir, assignment.model_name)

        url = job.camera_stream
        self._frame_reader = FrameReader(worker_key, url)

        pt_cfg = model_config.get("pt", {})
        self._attach_scheduler(
            model_path,
            build_detector=lambda: PersonPtDetector(model_path, model_config, assignment.model_name),
            target_fps=float(pt_cfg.get("target_fps", 12.0)),
            process_interval=float(pt_cfg.get("process_interval", 0.0)),
        )

    def _process_frame(self, frame, person_detections) -> None:
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
            upload_info = self._frame_uploader.upload_frame(self._worker_key, frame.frame_id, annotated)
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

        # Video only records while a person is actually in frame, with
        # the same boxes already drawn above for the upload.
        if person_detections:
            self._video_manager.feed_frame(annotated)

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
            self._face_controller.release()
        except Exception:
            self._log.exception("error releasing face controller")
        try:
            self._video_manager.close()
        except Exception:
            self._log.exception("error closing video manager")
