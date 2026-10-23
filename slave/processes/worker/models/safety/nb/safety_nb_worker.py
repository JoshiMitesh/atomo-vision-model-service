"""Standalone Safety `.nb` worker — a jobs.yaml entry with model_type:
"safety", entirely independent of any Person job.

Uses the shared SchedulerBackedWorker mixin
(runtimes/common/round_robin_scheduler.py) -- every job using the same
safety.nb file shares one detector/NPU handle and takes turns.
start()/stop()/is_alive()/the detections callback are all inherited
from the mixin.

Also attaches a face_controller (worker_factory.py wires this in for
both "person" and "safety" model types) — Face auto-triggers off this
model's own "person" class, same as safety_pt_worker.py.
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
from slave.processes.worker.models.safety.nb.safety_nb_detector import SafetyNbDetector
from slave.processes.worker.worker_factory import WorkerIdentity
from slave.runtimes.common.round_robin_scheduler import SchedulerBackedWorker
from slave.utils.time_utils import ist_now_iso


class SafetyNbWorker(SchedulerBackedWorker):
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

        library_file = model_config.get("nb", {}).get("library_file", "")
        model_path = os.path.join(models_dir, assignment.model_name)
        library_path = os.path.join(models_dir, library_file) if library_file else None

        url = job.camera_stream
        self._frame_reader = FrameReader(worker_key, url)

        nb_cfg = model_config.get("nb", {})
        self._attach_scheduler(
            model_path,
            build_detector=lambda: SafetyNbDetector(
                model_path, library_path, model_config, assignment.model_name
            ),
            target_fps=float(nb_cfg.get("target_fps", 12.0)),
            process_interval=float(nb_cfg.get("process_interval", 0.0)),
        )

    def _process_frame(self, frame, safety_detections) -> None:
        person_only = [d for d in safety_detections if d.class_name == "person"]
        face_detections = self._face_controller.update(frame.image, person_only)
        detections = safety_detections + face_detections

        object_key = url = None
        annotated = None
        if detections:
            annotated = draw_detections(frame.image, detections)
            upload_info = self._frame_uploader.upload_frame(self._worker_key, frame.frame_id, annotated)
            if upload_info is not None:
                object_key, url = upload_info

        width = height = None
        if object_key is not None:
            height, width = (int(v) for v in frame.image.shape[:2])

        # Every safety class (boots, no_vest, hardhat, person, ...) carries the
        # safety assignment's id; face detections riding along keep the
        # face assignment's own id.
        telemetry = group_detections(
            safety_detections, self._job, tracking_id=object_key,
            assign_id=self._job.camera_model_assign_id_for("safety"),
            model_id=self._job.model_id_for("safety"),
            model_name=self._job.model_name_for("safety"),
        ) + group_detections(face_detections, self._job, tracking_id=object_key)

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

        if safety_detections:
            self._video_manager.feed_frame(annotated)

    def _cleanup(self) -> None:
        try:
            self._frame_reader.stop()
        except Exception:
            self._log.exception("error stopping frame reader")
        # NOTE: the detector/NPU handle is NOT released here -- it is
        # shared across every job using this model_path. unregister_job()
        # (called before _cleanup) already releases it, but only once
        # this was the last job using that model_path.
        try:
            self._face_controller.release()
        except Exception:
            self._log.exception("error releasing face controller")
        try:
            self._video_manager.close()
        except Exception:
            self._log.exception("error closing video manager")
