"""Standalone Safety `.nb` worker — a jobs.yaml entry with model_type:
"safety", entirely independent of any Person job.

NOTE — NPU sharing: person_nb_worker.py's NpuRoundRobinScheduler exists
because the A311D NPU can only run one model instance at a time, and
every Person `.nb` job shares one scheduler/handle for that reason.
This worker does NOT participate in that scheduler (same as
face_nb_worker.py) — it owns an independent NbRuntime handle. If a
standalone Safety `.nb` job and a Person (or Face) `.nb` job ever run
concurrently on the same board, they will independently try to drive
the NPU, which may contend/fail depending on the A311D SDK's actual
behavior with multiple simultaneous handles. Generalizing the
round-robin scheduler to cover all `.nb` model types is the correct
fix if that combination needs to be supported.

Also attaches a face_controller (worker_factory.py wires this in for
both "person" and "safety" model types) — Face auto-triggers off this
model's own "person" class, same as safety_pt_worker.py.
"""

from __future__ import annotations

import os
import threading

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
from slave.utils.time_utils import ist_now_iso

FRAME_POLL_TIMEOUT_SECONDS = 1.0


class SafetyNbWorker:
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
        self._detector = SafetyNbDetector(model_path, library_path, model_config, assignment.model_name)

        url = job.camera_stream
        self._frame_reader = FrameReader(worker_key, url)

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"safety-nb-{worker_key}", daemon=True)

    def start(self) -> None:
        self._frame_reader.start()
        self._thread.start()
        self._log.info("safety .nb worker started")

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=10)
        self._cleanup()
        self._log.info("safety .nb worker stopped")

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
        safety_detections = self._detector.detect(frame.image)
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
        ) + group_detections(face_detections, self._job, tracking_id=object_key)

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

        if safety_detections:
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
