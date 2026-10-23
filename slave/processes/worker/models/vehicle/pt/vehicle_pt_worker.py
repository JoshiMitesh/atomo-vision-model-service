"""Standalone Vehicle `.pt` worker — a jobs.yaml entry with model_type:
"vehicle", entirely independent of any Person/Safety job. Same shape as
fire_pt_worker.py's frame loop, minus the severity post-process (that's
fire-specific) and minus a face_controller (worker_factory.py only
attaches one to "person" and "safety" model types — Vehicle has no
reason to trigger Face).

Uses the shared SchedulerBackedWorker mixin
(runtimes/common/round_robin_scheduler.py) -- every job using the same
vehicle.pt file shares one detector instance and takes turns.
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
from slave.processes.worker.models.vehicle.pt.vehicle_pt_detector import VehiclePtDetector
from slave.processes.worker.worker_factory import WorkerIdentity
from slave.runtimes.common.round_robin_scheduler import SchedulerBackedWorker
from slave.utils.time_utils import ist_now_iso


class VehiclePtWorker(SchedulerBackedWorker):
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

        model_path = os.path.join(models_dir, assignment.model_name)

        url = job.camera_stream
        self._frame_reader = FrameReader(worker_key, url)

        pt_cfg = model_config.get("pt", {})
        self._attach_scheduler(
            model_path,
            build_detector=lambda: VehiclePtDetector(model_path, model_config, assignment.model_name),
            target_fps=float(pt_cfg.get("target_fps", 12.0)),
            process_interval=float(pt_cfg.get("process_interval", 0.0)),
        )

    def _process_frame(self, frame, vehicle_detections) -> None:
        object_key = url = None
        annotated = None
        if vehicle_detections:
            annotated = draw_detections(frame.image, vehicle_detections)
            upload_info = self._frame_uploader.upload_frame(self._worker_key, frame.frame_id, annotated)
            if upload_info is not None:
                object_key, url = upload_info

        width = height = None
        if object_key is not None:
            height, width = (int(v) for v in frame.image.shape[:2])

        # Every vehicle class (car, motorcycle, bus, truck, other) shares
        # the one vehicle assignment's id/model_id/model_name — their
        # class names never equal model_type "vehicle", so the per-class
        # lookup in group_detections() would otherwise return null for
        # all of them (same fix as safety_pt_worker.py's "safety" override).
        telemetry = group_detections(
            vehicle_detections, self._job, tracking_id=object_key,
            assign_id=self._job.camera_model_assign_id_for("vehicle"),
            model_id=self._job.model_id_for("vehicle"),
            model_name=self._job.model_name_for("vehicle"),
        )

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

        if vehicle_detections:
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
            self._video_manager.close()
        except Exception:
            self._log.exception("error closing video manager")
