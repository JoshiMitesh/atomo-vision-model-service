"""Standalone Face `.nb` worker — a jobs.yaml entry with model_type:
"face", entirely independent of any Person job.

NOTE — NPU sharing: person_nb_worker.py's NpuRoundRobinScheduler exists
because the A311D NPU can only run one model instance at a time, and
every Person `.nb` job shares one scheduler/handle for that reason.
This worker does NOT participate in that scheduler — it owns an
independent NbRuntime handle, same as the shared scheduler used to
before it existed. If a standalone Face `.nb` job and a Person `.nb`
job ever run concurrently on the same board, both will try to drive
the NPU independently, which may contend/fail depending on the actual
A311D SDK's behavior with two simultaneous handles. Generalizing the
round-robin scheduler to cover both model types is the correct fix if
that combination needs to be supported — out of scope here since only
standalone (Face-only) operation was requested.
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
from slave.processes.worker.face.face_crop import crop_face
from slave.processes.worker.models.face.embedding.embedding_detector import with_embedding
from slave.processes.worker.models.face.nb.face_nb_detector import FaceNbDetector
from slave.processes.worker.worker_factory import WorkerIdentity
from slave.utils.time_utils import ist_now_iso

FRAME_POLL_TIMEOUT_SECONDS = 1.0


class FaceNbWorker:
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

        library_file = model_config.get("nb", {}).get("library_file", "")
        model_path = os.path.join(models_dir, assignment.model_name)
        library_path = os.path.join(models_dir, library_file) if library_file else None
        self._detector = with_embedding(
            FaceNbDetector(model_path, library_path, model_config, assignment.model_name), model_config, models_dir
        )

        url = job.camera_stream
        self._frame_reader = FrameReader(worker_key, url)

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"face-nb-{worker_key}", daemon=True)

    def start(self) -> None:
        self._frame_reader.start()
        self._thread.start()
        self._log.info("standalone face .nb worker started")

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=10)
        self._cleanup()
        self._log.info("standalone face .nb worker stopped")

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

        # Upload one cropped photo per detected face (not the whole
        # camera frame). media.object_key/url/width/height describe the
        # first crop; media.crops lists every saved crop.
        object_key = url = None
        width = height = None
        crops = None
        for index, detection in enumerate(detections):
            crop = crop_face(frame.image, detection.bbox)
            if crop is None:
                continue
            upload_info = self._frame_uploader.upload_face_crop(
                self._worker_key, frame.frame_id, f"face_{index}", crop
            )
            if upload_info is None:
                continue
            crop_height, crop_width = (int(v) for v in crop.shape[:2])
            if crops is None:
                crops = []
                object_key, url = upload_info
                width, height = crop_width, crop_height
            crops.append(
                {
                    "object_key": upload_info[0],
                    "url": upload_info[1],
                    "width": crop_width,
                    "height": crop_height,
                }
            )

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
            crops=crops,
        )
        self._detection_publisher.publish_detection(result)

        if detections:
            self._video_manager.feed_frame(draw_detections(frame.image, detections))

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
