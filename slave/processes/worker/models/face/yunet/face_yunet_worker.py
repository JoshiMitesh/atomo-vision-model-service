"""Standalone Face YuNet worker.

YuNet is selected from jobs.yaml exactly like the PT/NB face models.
The event/crop/tracking behavior intentionally matches the existing
standalone face .pt worker:
- FaceController provides camera-local tracking.
- Only a new face track creates a face event and uploads a crop.
- The uploaded image is the padded face crop, not the full camera frame.
- Existing recognition/cooldown handling remains unchanged.
SFace remains internal to the embedding wrapper.
"""

from __future__ import annotations

import os
import threading
import time

import cv2

from slave.processes.worker.models.face.face_preprocess import crop_face, resize_reference_frame

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
from slave.processes.worker.models.face.embedding.embedding_detector import with_embedding
from slave.processes.worker.models.face.yunet.face_yunet_detector import FaceYunetDetector
from slave.processes.worker.worker_factory import WorkerIdentity
from slave.utils.time_utils import ist_now_iso

FRAME_POLL_TIMEOUT_SECONDS = 1.0


class FaceYunetWorker:
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
        self._detector = with_embedding(
            FaceYunetDetector(model_path, model_config, assignment.model_name),
            model_config,
            models_dir,
            yunet_model_name=assignment.model_name,
        )

        self._face_tracker = FaceController(
            job_id=job.job_id,
            detector_factory=lambda: self._detector,
            off_timeout_sec=0,
            enabled=True,
        )

        self._frame_reader = FrameReader(worker_key, job.camera_stream)

        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"face-yunet-{worker_key}",
            daemon=True,
        )

    def start(self) -> None:
        self._frame_reader.start()
        self._thread.start()
        self._log.info("standalone face YuNet worker started")

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=10)
        self._cleanup()
        self._log.info("standalone face YuNet worker stopped")

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
        process_start = time.monotonic()
        frame_age = max(0.0, time.time() - frame.timestamp)

        # Match the working project: face inference/crops operate on 960x540.
        timing = {}

        stage_start = time.monotonic()
        working_image = resize_reference_frame(frame.image)
        timing["resize"] = time.monotonic() - stage_start

        stage_start = time.monotonic()
        detections = self._detector.detect(working_image)
        timing["detect_and_embedding"] = time.monotonic() - stage_start

        detections = [
            detection for detection in detections
            if detection.confidence >= 0.60
            and (detection.bbox[2] - detection.bbox[0]) >= 50.0
            and (detection.bbox[3] - detection.bbox[1]) >= 50.0
            and bool(detection.extra.get("face_visible", True))
        ]

        stage_start = time.monotonic()
        self._face_tracker.track_detections(detections)
        timing["tracking"] = time.monotonic() - stage_start

        event_count = 0
        for detection, track_id in self._face_tracker.new_face_events:
            event_count += 1

            stage_start = time.monotonic()
            crop = self._crop_face(working_image, detection.bbox)
            timing["crop"] = timing.get("crop", 0.0) + (time.monotonic() - stage_start)
            if crop is None:
                continue

            stage_start = time.monotonic()
            upload_info = self._frame_uploader.upload_face_crop(
                self._worker_key,
                frame.frame_id,
                track_id,
                crop,
            )
            timing["minio"] = timing.get("minio", 0.0) + (time.monotonic() - stage_start)
            if upload_info is None:
                continue

            object_key, url = upload_info
            crop_height, crop_width = (int(v) for v in crop.shape[:2])
            telemetry = group_detections(
                [detection],
                self._job,
                tracking_id=track_id,
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
                telemetry=telemetry,
                object_key=object_key,
                url=url,
                width=crop_width,
                height=crop_height,
            )

            stage_start = time.monotonic()
            self._detection_publisher.publish_detection(result)
            timing["publish"] = timing.get("publish", 0.0) + (time.monotonic() - stage_start)

        if detections:
            stage_start = time.monotonic()
            self._video_manager.feed_frame(draw_detections(working_image, detections))
            timing["video_queue"] = time.monotonic() - stage_start

        total = time.monotonic() - process_start

        if event_count or frame_age >= 1.0 or total >= 1.0:
            self._log.info(
                "FACE_TIMING frame=%s age=%.3fs resize=%.3fs detect_embedding=%.3fs "
                "tracking=%.3fs crop=%.3fs minio=%.3fs publish=%.3fs video_queue=%.3fs "
                "total=%.3fs events=%d",
                frame.frame_id,
                frame_age,
                timing.get("resize", 0.0),
                timing.get("detect_and_embedding", 0.0),
                timing.get("tracking", 0.0),
                timing.get("crop", 0.0),
                timing.get("minio", 0.0),
                timing.get("publish", 0.0),
                timing.get("video_queue", 0.0),
                total,
                event_count,
            )

    _crop_face = staticmethod(crop_face)

    def _cleanup(self) -> None:
        try:
            self._frame_reader.stop()
        except Exception:
            self._log.exception("error stopping frame reader")
        try:
            self._face_tracker.release()
        except Exception:
            self._log.exception("error releasing face tracker")
        try:
            self._detector.release()
        except Exception:
            self._log.exception("error releasing detector")
        try:
            self._video_manager.close()
        except Exception:
            self._log.exception("error closing video manager")
