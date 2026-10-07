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

        # Keep the exact tracking/event behavior used by the existing
        # standalone .pt face worker. Tracking is camera/job-local and a
        # continuously visible face creates only one new-face event until
        # its track expires.
        self._face_tracker = FaceController(
            job_id=job.job_id,
            detector_factory=lambda: self._detector,
            off_timeout_sec=0,
            enabled=True,
        )

        self._frame_reader = FrameReader(worker_key, job.camera_stream)

        self._stop_event = threading.Event()
        self._last_live_publish = 0.0
        self._live_publish_interval = 0.20

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
        # YuNet detections are the event source. Do not apply an additional
        # face-quality/landmark gate here: every detector hit is a face event.
        detections = self._detector.detect(frame.image)
        quality_detections = detections

        tracked_detections = self._face_tracker.track_detections_with_ids(detections)

        # Keep the live UI overlay current without turning every video frame
        # into a persisted Master event or an email alert.
        now = time.monotonic()
        if quality_detections and now - self._last_live_publish >= self._live_publish_interval:
            live_telemetry = group_detections(
                quality_detections,
                self._job,
                tracking_id=(
                    tracked_detections[0][1] if len(tracked_detections) == 1 else None
                ),
            )
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
                object_key=None,
                url=None,
                width=None,
                height=None,
            )
            self._detection_publisher.publish_live_detection(
                live_result,
                self._job.camera_id,
            )
            self._last_live_publish = now

        # Publish/upload only when a face track first appears. A continuously
        # visible face therefore produces one event and one MinIO face crop.
        # Emit detector hits on every processed frame. The stream reader
        # already samples the camera, so this produces one event per sampled
        # frame rather than tracker-only first sightings.
        for detection, track_id in tracked_detections:
            # Publishing the detection must not depend on MinIO succeeding.
            # A temporary crop-upload failure should remove the image, not
            # the face event itself.
            object_key = None
            url = None
            crop_width = None
            crop_height = None

            crop = self._crop_face(frame.image, detection.bbox)
            if crop is not None:
                try:
                    upload_info = self._frame_uploader.upload_face_crop(
                        self._worker_key,
                        frame.frame_id,
                        track_id,
                        crop,
                    )
                    if upload_info is not None:
                        object_key, url = upload_info
                        crop_height, crop_width = (int(v) for v in crop.shape[:2])
                    else:
                        self._log.warning(
                            "face crop upload returned no result; publishing detection "
                            "without media frame_id=%s track_id=%s",
                            frame.frame_id,
                            track_id,
                        )
                except Exception:
                    self._log.exception(
                        "face crop upload failed; publishing detection without media "
                        "frame_id=%s track_id=%s",
                        frame.frame_id,
                        track_id,
                    )
            else:
                self._log.warning(
                    "face crop was invalid; publishing detection without media "
                    "frame_id=%s track_id=%s",
                    frame.frame_id,
                    track_id,
                )

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
                latitude=self._job.latitude,
                longitude=self._job.longitude,
                telemetry=telemetry,
                object_key=object_key,
                url=url,
                width=crop_width,
                height=crop_height,
            )
            self._detection_publisher.publish_detection(result)

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
