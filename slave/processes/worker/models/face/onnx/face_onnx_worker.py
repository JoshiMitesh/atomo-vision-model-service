"""TEMPORARY TEST worker — exact copy of face_pt_worker.py, swapped to
FaceOnnxDetector so a raw .onnx file can run as job_001's actual face
detector. Not permanent — see face_onnx_detector.py's docstring for
what to revert when testing is done.
"""

from __future__ import annotations

import os
import threading
import time

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
from slave.processes.worker.models.face.onnx.face_onnx_detector import FaceOnnxDetector
from slave.processes.worker.models.face.face_tracker import DEFAULT_IOU_THRESHOLD, DEFAULT_STALE_SECONDS, FaceTracker
from slave.processes.worker.worker_factory import WorkerIdentity
from slave.utils.time_utils import ist_now_iso
from slave.profiling.profiler import Profiler
from slave.processes.publisher.profiling_publisher import ProfilingPublisher

FRAME_POLL_TIMEOUT_SECONDS = 1.0


class FaceOnnxWorker:
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
        profiling_publisher: ProfilingPublisher,
    ) -> None:
        self._job = job
        self._worker_key = worker_key
        self._detection_publisher = detection_publisher
        self._frame_uploader = frame_uploader
        self._video_manager = video_manager
        self._identity = identity
        self._log = get_logger(__name__, job_id=job.job_id)
        self._profiling_publisher = profiling_publisher
        self._profiler = Profiler(model_name=assignment.model_name)

        model_path = os.path.join(models_dir, assignment.model_name)
        self._detector = with_embedding(
            FaceOnnxDetector(model_path, model_config, assignment.model_name), model_config, models_dir
        )

        url = job.camera_stream
        self._frame_reader = FrameReader(worker_key, url)

        self._crop_padding = model_config.get("crop_padding", 0.25)
        self._face_tracker = FaceTracker(
            model_config.get("track_iou_threshold", DEFAULT_IOU_THRESHOLD),
            model_config.get("track_stale_seconds", DEFAULT_STALE_SECONDS),
        )

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"face-onnx-{worker_key}", daemon=True)

    def start(self) -> None:
        self._frame_reader.start()
        self._thread.start()
        self._log.info("standalone face .onnx (TEST) worker started")

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=10)
        self._cleanup()
        self._log.info("standalone face .onnx (TEST) worker stopped")

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
        self._profiler.start()
        if hasattr(frame, 'timestamp'):
            self._profiler.record_queue_wait((time.time() - frame.timestamp) * 1000.0)
        detections = self._detector.detect(frame.image)
        new_faces = self._face_tracker.update(detections)

        object_key = url = None
        annotated = None
        crops = None
        width = height = None
        if detections:
            annotated = draw_detections(frame.image, detections)
            uploaded = self._frame_uploader.upload_face_crops(
                self._worker_key, frame.frame_id, frame.image, new_faces,
                self._crop_padding,
            ) if new_faces else []
            if uploaded:
                object_key, url, _, width, height = uploaded[0]
                crops = [
                    {"object_key": k, "url": u, "bbox": list(b), "width": w, "height": h}
                    for k, u, b, w, h in uploaded
                ]

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
            telemetry=telemetry,
            object_key=object_key,
            url=url,
            width=width,
            height=height,
            crops=crops,
        )
        self._profiler.stop()
        self._profiling_publisher.publish(self._profiler.report())
        self._detection_publisher.publish_detection(result)

        if detections:
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
            self._video_manager.close()
        except Exception:
            self._log.exception("error closing video manager")
