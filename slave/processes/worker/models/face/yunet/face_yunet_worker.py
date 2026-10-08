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

from dataclasses import replace
import os
import threading

import cv2

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
from slave.processes.worker.face.face_quality import passes_face_quality_gate
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
        detections = self._detector.detect(frame.image)
        quality_detections = [
            detection
            for detection in detections
            if passes_face_quality_gate(detection)
        ]

        # Publish lightweight face boxes on every frame for the live overlay.
        # Do not attach embeddings/media here: these messages are transport-only
        # live detection and must never create/learn a recognition profile.
        if detections:
            live_detections = [
                replace(detection, extra={})
                for detection in detections
            ]
            live_telemetry = group_detections(
                live_detections,
                self._job,
                tracking_id=f"face_live:{self._job.camera_id}",
            )
            live_telemetry = [
                replace(telemetry, event_type="face_detected")
                for telemetry in live_telemetry
            ]
            live_result = DetectionResult(
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
                telemetry=live_telemetry,
                width=int(frame.image.shape[1]),
                height=int(frame.image.shape[0]),
            )
            self._detection_publisher.publish_detection(live_result)

        # Recognition/profile work is handled separately from the live
        # detection stream. Every detected face is sent to the live overlay;
        # no tracker is used to decide whether a bbox is published.

        if detections:
            self._video_manager.feed_frame(draw_detections(frame.image, detections))

    @staticmethod
    def _crop_face(frame_image, bbox):
        height, width = frame_image.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(width, int(x1)))
        y1 = max(0, min(height, int(y1)))
        x2 = max(0, min(width, int(x2)))
        y2 = max(0, min(height, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return None

        # Match the existing reference face-crop behavior:
        # 60% padding around the detector box, minimum 150 px, CLAHE,
        # and JPEG quality 90 at upload time.
        box_w = x2 - x1
        box_h = y2 - y1
        pad_w = int(box_w * 0.60)
        pad_h = int(box_h * 0.60)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(width, x2 + pad_w)
        y2 = min(height, y2 + pad_h)

        crop = frame_image[y1:y2, x1:x2].copy()
        crop_h, crop_w = crop.shape[:2]
        min_size = 150
        if crop_h < min_size or crop_w < min_size:
            scale_factor = max(min_size / crop_w, min_size / crop_h)
            crop = cv2.resize(
                crop,
                (int(crop_w * scale_factor), int(crop_h * scale_factor)),
                interpolation=cv2.INTER_CUBIC,
            )

        try:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            crop = cv2.cvtColor(
                cv2.merge([clahe.apply(l), a, b]),
                cv2.COLOR_LAB2BGR,
            )
        except Exception:
            pass

        return crop

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
            self._video_manager.close()
        except Exception:
            self._log.exception("error closing video manager")
