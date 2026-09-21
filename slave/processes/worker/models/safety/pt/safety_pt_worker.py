"""Safety `.pt` worker — job lifecycle, frame loop, result delivery.

Same shape as person_pt_worker.py: this Safety job also gets a
face_controller (worker_factory.py attaches one to both "person" and
"safety" model types) — but the trigger signal is filtered down to
just this model's own "person" class, not the full 9-class detection
list, so Face only activates when Safety actually sees a person in
frame (see face_controller.update()'s truthiness check).
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
from slave.processes.worker.models.safety.pt.safety_pt_detector import SafetyPtDetector
from slave.processes.worker.worker_factory import WorkerIdentity
from slave.utils.time_utils import ist_now_iso

FRAME_POLL_TIMEOUT_SECONDS = 1.0


class SafetyPtWorker:
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
        self._detector = SafetyPtDetector(model_path, model_config, assignment.model_name)

        url = job.camera_stream
        self._frame_reader = FrameReader(worker_key, url)

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"safety-pt-{worker_key}", daemon=True)

    def start(self) -> None:
        self._frame_reader.start()
        self._thread.start()
        self._log.info("safety .pt worker started")

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=10)
        self._cleanup()
        self._log.info("safety .pt worker stopped")

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
        # Face only triggers off Safety's own "person" class, not any
        # of the other 8 PPE-compliance classes.
        person_only = [d for d in safety_detections if d.class_name == "person"]
        face_detections = self._face_controller.update(frame.image, person_only)
        detections = safety_detections + face_detections

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

        # Video records whenever Safety produced any detection this
        # frame (mirrors Person's "record while a person is in frame"
        # rule, generalized to "record while anything Safety-relevant
        # is in frame").
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
