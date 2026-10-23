"""Standalone Face `.nb` worker — a jobs.yaml entry with model_type:
"face", entirely independent of any Person job.

Uses the shared SchedulerBackedWorker mixin
(runtimes/common/round_robin_scheduler.py) -- the A311D NPU can only
run one model instance at a time, so every job using the same
face.nb file (standalone Face jobs, here) shares one detector/handle
and takes turns. start()/stop()/is_alive()/the detections callback are
all inherited from the mixin.
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
from slave.processes.worker.face.face_crop import crop_face
from slave.processes.worker.models.face.embedding.embedding_detector import with_embedding
from slave.processes.worker.models.face.nb.face_nb_detector import FaceNbDetector
from slave.processes.worker.worker_factory import WorkerIdentity
from slave.runtimes.common.round_robin_scheduler import SchedulerBackedWorker
from slave.utils.time_utils import ist_now_iso


class FaceNbWorker(SchedulerBackedWorker):
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

        url = job.camera_stream
        self._frame_reader = FrameReader(worker_key, url)

        nb_cfg = model_config.get("nb", {})
        self._attach_scheduler(
            model_path,
            build_detector=lambda: with_embedding(
                FaceNbDetector(model_path, library_path, model_config, assignment.model_name),
                model_config,
                models_dir,
            ),
            target_fps=float(nb_cfg.get("target_fps", 12.0)),
            process_interval=float(nb_cfg.get("process_interval", 0.0)),
        )

    def _process_frame(self, frame, detections) -> None:
        # Upload one cropped photo per detected face (not the whole
        # camera frame). media.object_key/url/width/height describe the
        # first crop; media.crops lists every saved crop.
        object_key = url = None
        width = height = None
        crops = None
        for index, detection in enumerate(detections):
            crop = crop_face(frame.image, detection.bbox)
            if crop is None:
                continue
            upload_info = self._frame_uploader.upload_face_crop(
                self._worker_key, frame.frame_id, f"face_{index}", crop
            )
            if upload_info is None:
                continue
            crop_height, crop_width = (int(v) for v in crop.shape[:2])
            if crops is None:
                crops = []
                object_key, url = upload_info
                width, height = crop_width, crop_height
            crops.append(
                {
                    "object_key": upload_info[0],
                    "url": upload_info[1],
                    "width": crop_width,
                    "height": crop_height,
                }
            )

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
            crops=crops,
        )
        self._detection_publisher.publish_detection(result)

        if detections:
            self._video_manager.feed_frame(draw_detections(frame.image, detections))

    def _cleanup(self) -> None:
        try:
            self._frame_reader.stop()
        except Exception:
            self._log.exception("error stopping frame reader")
        # NOTE: the detector/NPU handle is NOT released here -- it is
        # shared across every job using this model_path, released by
        # the scheduler registry only once the last job unregisters.
        try:
            self._video_manager.close()
        except Exception:
            self._log.exception("error closing video manager")
