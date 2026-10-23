"""Standalone Face `.pt` worker — a jobs.yaml entry with model_type:
"face", entirely independent of any Person job.

Unlike face_controller.py's person-chained Face (only active while a
person is in frame, on the Person job's own stream), this worker owns
its own RTSP stream/FrameReader and runs Face detection on every
frame, unconditionally, for the lifetime of its own job.

Uses the shared SchedulerBackedWorker mixin
(runtimes/common/round_robin_scheduler.py) -- every job using the same
face.pt file shares one detector instance and takes turns, same as
PersonPtWorker. start()/stop()/is_alive()/the detections callback are
all inherited from the mixin.
"""

from __future__ import annotations

import os

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
from slave.processes.worker.models.face.embedding.embedding_detector import with_embedding
from slave.processes.worker.models.face.pt.face_pt_detector import FacePtDetector
from slave.processes.worker.face.face_controller import FaceController
from slave.processes.worker.face.face_quality import passes_face_quality_gate
from slave.processes.worker.worker_factory import WorkerIdentity
from slave.runtimes.common.round_robin_scheduler import SchedulerBackedWorker
from slave.utils.time_utils import ist_now_iso


class FacePtWorker(SchedulerBackedWorker):
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
            build_detector=lambda: with_embedding(
                FacePtDetector(model_path, model_config, assignment.model_name), model_config, models_dir
            ),
            target_fps=float(pt_cfg.get("target_fps", 12.0)),
            process_interval=float(pt_cfg.get("process_interval", 0.0)),
        )
        # Never person-gated here (track_detections() only, not
        # update()), so _activate()/detector_factory is never actually
        # invoked -- kept only so release() stays a correct no-op.
        self._face_tracker = FaceController(
            job_id=job.job_id,
            detector_factory=lambda: self._scheduler._detector,
            off_timeout_sec=0,
            enabled=True,
        )

    def _process_frame(self, frame, detections) -> None:
        # Match the shared face-quality filters before tracking/event
        # creation. Keep the existing embedding visibility check so
        # partially occluded faces remain excluded for standalone PT.
        quality_detections = []
        for detection in detections:
            if not passes_face_quality_gate(detection):
                continue
            if not bool(detection.extra.get("face_visible", False)):
                continue
            quality_detections.append(detection)

        self._face_tracker.track_detections(quality_detections)

        # Publish/upload only when a face track first appears. A continuously
        # visible face therefore produces one event and one MinIO object.
        for detection, track_id in self._face_tracker.new_face_events:
            crop = self._crop_face(frame.image, detection.bbox)
            if crop is None:
                continue

            upload_info = self._frame_uploader.upload_face_crop(
                self._worker_key,
                frame.frame_id,
                track_id,
                crop,
            )
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
        if x2 <= x1 or y2 <= y1:
            return None

        # Match the reference atomo-ai-mitesh crop behavior:
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
        # NOTE: the shared detector is NOT released here -- unregister_job()
        # (called before _cleanup) already releases it, but only once
        # this was the last job using that model_path.
        try:
            self._video_manager.close()
        except Exception:
            self._log.exception("error closing video manager")
