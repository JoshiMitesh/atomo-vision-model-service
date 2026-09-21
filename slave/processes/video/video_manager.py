"""Per-job facade tying VideoRecorder to MinIO video uploads.

Segment uploads run on a small background executor so the frame loop
never blocks on network I/O. A video failure is caught and logged here
and must never propagate into worker/job-store logic (CLAUDE.md §18).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from slave.logging.logger import get_logger
from slave.processes.minio.video_uploader import VideoUploader
from slave.processes.video.video_recorder import VideoRecorder

log = get_logger(__name__)


class VideoManager:
    def __init__(
        self,
        job_id: str,
        video_uploader: VideoUploader,
        temp_dir: str,
        segment_seconds: int,
        fps: int,
    ) -> None:
        self._job_id = job_id
        self._video_uploader = video_uploader
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"video-upload-{job_id}")
        self._recorder = VideoRecorder(job_id, temp_dir, segment_seconds, fps, self._on_segment_ready)

    def feed_frame(self, frame: np.ndarray) -> None:
        try:
            self._recorder.feed_frame(frame)
        except Exception:
            log.exception("video recording failed for job_id=%s (frame dropped from video)", self._job_id)

    def _on_segment_ready(self, local_path: str, segment_name: str) -> None:
        self._executor.submit(self._upload_segment, local_path, segment_name)

    def _upload_segment(self, local_path: str, segment_name: str) -> None:
        try:
            self._video_uploader.upload_video_segment(self._job_id, local_path, segment_name)
        except Exception:
            log.exception("video segment upload failed for job_id=%s", self._job_id)

    def close(self) -> None:
        """Finish current segment, upload it, then release resources
        (graceful shutdown, CLAUDE.md §21)."""
        try:
            self._recorder.close()
        except Exception:
            log.exception("failed to flush final video segment for job_id=%s", self._job_id)
        self._executor.shutdown(wait=True)
