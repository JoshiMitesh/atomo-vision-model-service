"""Person `.pt` worker — job lifecycle, frame loop, result delivery.

Owns everything specific to running one job: its own RTSP reader,
detector instance, stop event, and video/upload state. Never shares
state with another job's worker (CLAUDE.md §20).
"""

from __future__ import annotations

import os
import threading

from slave.logging.logger import get_logger
from slave.models.job import JobConfig
from slave.processes.detection.detection_result import DetectionResult
from slave.processes.detection.frame_annotator import draw_detections
from slave.processes.minio.frame_uploader import FrameUploader
from slave.processes.publisher.detection_publisher import DetectionPublisher
from slave.processes.stream.frame_reader import FrameReader
from slave.processes.stream.stream_connector import build_stream_url
from slave.processes.video.video_manager import VideoManager
from slave.processes.worker.models.person.pt.person_pt_detector import PersonPtDetector

FRAME_POLL_TIMEOUT_SECONDS = 1.0


class PersonPtWorker:
    def __init__(
        self,
        job: JobConfig,
        model_config: dict,
        models_dir: str,
        detection_publisher: DetectionPublisher,
        frame_uploader: FrameUploader,
        video_manager: VideoManager,
    ) -> None:
        self._job = job
        self._detection_publisher = detection_publisher
        self._frame_uploader = frame_uploader
        self._video_manager = video_manager
        self._log = get_logger(__name__, job_id=job.job_id)

        model_path = os.path.join(models_dir, job.model_name)
        self._detector = PersonPtDetector(model_path, model_config)

        url = build_stream_url(job.master_ip, job.stream_port, job.stream_name)
        self._frame_reader = FrameReader(job.job_id, url)

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"person-pt-{job.job_id}", daemon=True)

    def start(self) -> None:
        self._frame_reader.start()
        self._thread.start()
        self._log.info("person .pt worker started")

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=10)
        self._cleanup()
        self._log.info("person .pt worker stopped")

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
        result = DetectionResult(job_id=self._job.job_id, frame_id=frame.frame_id, detections=detections)
        self._detection_publisher.publish_detection(result)

        if result.has_detections:
            annotated = draw_detections(frame.image, detections)
            self._frame_uploader.upload_frame(self._job.job_id, frame.frame_id, annotated)

        self._video_manager.feed_frame(frame.image)

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
