"""Person `.nb` worker — job lifecycle around the shared
SchedulerBackedWorker mixin (runtimes/common/round_robin_scheduler.py).

The worker no longer owns its own detector or polling thread: the NPU
can only run one model instance at a time, so all `.nb` jobs using the
same model file share one RoundRobinScheduler/detector and this
worker just registers a FrameReader + result callback with it (via
_attach_scheduler() -- start()/stop()/is_alive()/the callback wrapper
are all inherited from the mixin, not written here). Everything
downstream of detect() (face detection, upload, publish, video) still
runs per-job, on the scheduler thread, at that job's turn. The
scheduler itself is shared with every other model type's `.nb`/`.pt`
workers (e.g. Number-Plate) -- it has no Person-specific code.
"""

from __future__ import annotations

import os
import time

from slave.logging.logger import get_logger
from slave.models.detection import group_detections
from slave.models.frame import Frame
from slave.models.job import JobConfig, ModelAssignment
from slave.processes.detection.detection_result import DetectionResult
from slave.processes.detection.frame_annotator import draw_detections
from slave.processes.minio.frame_uploader import FrameUploader
from slave.processes.publisher.detection_publisher import DetectionPublisher
from slave.processes.stream.frame_reader import FrameReader
from slave.processes.video.video_manager import VideoManager
from slave.processes.worker.face.face_controller import FaceController
from slave.processes.worker.models.person.nb.person_nb_detector import PersonNbDetector
from slave.processes.worker.worker_factory import WorkerIdentity
from slave.runtimes.common.round_robin_scheduler import SchedulerBackedWorker
from slave.utils.time_utils import ist_now_iso


class PersonNbWorker(SchedulerBackedWorker):
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
        self._low_cpu_mode = bool(model_config.get("nb", {}).get("low_cpu_mode", True))
        self._publish_every_n_frames = max(1, int(model_config.get("nb", {}).get("publish_every_n_frames", 5)))
        self._minio_upload_interval = max(0.0, float(model_config.get("nb", {}).get("minio_upload_interval_seconds", 4.0)))
        self._last_minio_upload = 0.0
        self._frame_counter = 0

        library_file = model_config.get("nb", {}).get("library_file", "")
        model_path = os.path.join(models_dir, assignment.model_name)
        library_path = os.path.join(models_dir, library_file) if library_file else None

        url = job.camera_stream
        self._frame_reader = FrameReader(
            job.job_id,
            url,
            decode_every_n_frames=max(1, int(model_config.get("nb", {}).get("decode_every_n_frames", 3))),
        )

        nb_cfg = model_config.get("nb", {})
        self._attach_scheduler(
            model_path,
            build_detector=lambda: PersonNbDetector(
                model_path, library_path, model_config, assignment.model_name
            ),
            target_fps=float(nb_cfg.get("target_fps", 2.0)),
            process_interval=float(nb_cfg.get("process_interval", 0.0)),
        )

    def _process_frame(self, frame: Frame, person_detections: list) -> None:
        self._frame_counter += 1
        if self._frame_counter % self._publish_every_n_frames != 0:
            return

        face_detections = self._face_controller.update(frame.image, person_detections)
        detections = person_detections + face_detections

        # Keep the low-CPU cadence, but still upload sampled annotated
        # frames when this publish window is active. This avoids burning
        # CPU on every frame while still making detection images visible in
        # MinIO when a published frame actually contains detections.
        object_key = url = None
        width = height = None
        annotated = None
        upload_due = time.monotonic() - self._last_minio_upload >= self._minio_upload_interval
        if detections and upload_due:
            annotated = draw_detections(frame.image, detections)
            upload_info = self._frame_uploader.upload_frame(self._job.job_id, frame.frame_id, annotated)
            if upload_info is not None:
                object_key, url = upload_info
            self._last_minio_upload = time.monotonic()
        if object_key is not None:
            height, width = (int(v) for v in frame.image.shape[:2])

        # tracking_id is the frame's saved MinIO image id, not a real
        # per-object tracker. In low-CPU mode we still keep it null unless
        # the upload succeeded for this sampled frame.
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

        # Video recording is intentionally disabled in low-CPU mode because
        # the reference script does not spend its hot loop doing this work.
        if not self._low_cpu_mode and person_detections:
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
