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
from queue import Empty, Full, Queue
import os
import threading
import time

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
from slave.processes.worker.models.face.embedding.face_embedder import FaceEmbedder
from slave.processes.worker.models.face.yunet.face_yunet_detector import FaceYunetDetector
from slave.processes.worker.worker_factory import WorkerIdentity
from slave.utils.time_utils import ist_now_iso

FRAME_POLL_TIMEOUT_SECONDS = 1.0
FACE_DETECTION_FPS = 6.0
FACE_DETECTION_INTERVAL_SECONDS = 1.0 / FACE_DETECTION_FPS
RECOGNITION_INTERVAL_SECONDS = 2.0


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
        # Live detection deliberately uses the raw YuNet detector. Recognition
        # must never be allowed to filter live boxes through face_visible.
        self._detector = FaceYunetDetector(
            model_path, model_config, assignment.model_name
        )
        embedding_cfg = model_config.get("embedding", {})
        self._embedder = FaceEmbedder(
            embedding_cfg,
            models_dir,
            yunet_model_name=assignment.model_name,
        )

        self._frame_reader = FrameReader(worker_key, job.camera_stream)

        # Recognition events are rate-limited without using FaceController.
        # Live face_detected telemetry remains every frame.
        self._recognition_event_history: list[dict] = []
        self._next_detection_time = 0.0

        # Single-slot queue: recognition always receives the newest detected frame.
        self._recognition_queue: Queue = Queue(maxsize=1)
        self._recognition_thread = threading.Thread(
            target=self._recognition_run,
            name=f"face-yunet-recognition-{worker_key}",
            daemon=True,
        )

        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"face-yunet-{worker_key}",
            daemon=True,
        )

    def start(self) -> None:
        self._frame_reader.start()
        self._thread.start()
        self._recognition_thread.start()
        self._log.info("standalone face YuNet worker started (6 FPS detection, latest-frame recognition)")

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=10)
        self._recognition_thread.join(timeout=10)
        self._cleanup()
        self._log.info("standalone face YuNet worker stopped")

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            frame = self._frame_reader.get_latest_frame(timeout=FRAME_POLL_TIMEOUT_SECONDS)
            if frame is None:
                continue
            now = time.monotonic()
            if now < self._next_detection_time:
                continue
            self._next_detection_time = now + FACE_DETECTION_INTERVAL_SECONDS
            try:
                self._process_frame(frame)
            except Exception:
                self._log.exception("frame processing failed for frame_id=%s", frame.frame_id)

    def _process_frame(self, frame) -> None:
        # Raw YuNet detections are the source of truth for the live overlay.
        # Run this path at a fixed 6 FPS and never let recognition quality
        # filtering remove a visible face from the stream.
        detections = self._detector.detect(frame.image)

        # Recognition remains separate and rate-limited. Only when a face is
        # eligible for recognition do we run SFace/YuNet landmarks.
        # No brightness/contrast/visibility quality gate here.
        # Every YuNet detection is eligible for recognition; pose/angle
        # rejection is handled inside FaceEmbedder using facial landmarks.
        if detections:
            self._enqueue_latest_recognition(frame, detections)

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

        if detections:
            self._video_manager.feed_frame(draw_detections(frame.image, detections))

    def _enqueue_latest_recognition(self, frame, detections) -> None:
        item = (frame, detections)
        try:
            self._recognition_queue.put_nowait(item)
            return
        except Full:
            pass
        try:
            self._recognition_queue.get_nowait()
        except Empty:
            pass
        try:
            self._recognition_queue.put_nowait(item)
        except Full:
            pass

    def _recognition_run(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame, detections = self._recognition_queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                embedded = self._embedder.attach(frame.image, detections)
                now = time.monotonic()
                for detection in embedded:
                    if detection.extra.get("face_visible") is not True:
                        continue
                    if not self._should_publish_recognition_event(detection, now):
                        continue
                    crop = self._crop_face(frame.image, detection.bbox)
                    if crop is None:
                        continue
                    track_id = f"face_recognition:{self._job.camera_id}:{frame.frame_id}"
                    upload_info = self._frame_uploader.upload_face_crop(
                        self._worker_key, frame.frame_id, track_id, crop
                    )
                    if upload_info is None:
                        continue
                    object_key, url = upload_info
                    crop_height, crop_width = (int(v) for v in crop.shape[:2])
                    telemetry = group_detections([detection], self._job, tracking_id=track_id)
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
                        width=crop_width,
                        height=crop_height,
                    )
                    self._detection_publisher.publish_detection(result)
            except Exception:
                self._log.exception(
                    "latest-frame face recognition failed for frame_id=%s",
                    getattr(frame, "frame_id", None),
                )

    def _should_publish_recognition_event(self, detection, now: float) -> bool:
        x1, y1, x2, y2 = detection.bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Keep the same face from generating recognition work continuously.
        # A 2-second interval still lets recognition refresh while avoiding
        # one database/profile event per frame.
        interval = 2.0
        distance_px = 140.0

        for item in self._recognition_event_history:
            if now - item["time"] > 10.0:
                continue
            if item["class_name"] != detection.class_name:
                continue
            distance = ((cx - item["cx"]) ** 2 + (cy - item["cy"]) ** 2) ** 0.5
            if distance <= distance_px:
                if now - item["time"] < interval:
                    return False
                item.update(time=now, cx=cx, cy=cy)
                return True

        self._recognition_event_history.append(
            {"time": now, "cx": cx, "cy": cy, "class_name": detection.class_name}
        )
        if len(self._recognition_event_history) > 50:
            self._recognition_event_history = self._recognition_event_history[-50:]
        return True

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
            self._embedder.release()
        except Exception:
            self._log.exception("error releasing face embedder")
        try:
            self._video_manager.close()
        except Exception:
            self._log.exception("error closing video manager")
