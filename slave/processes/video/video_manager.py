"""Per-job facade tying VideoRecorder to MinIO video uploads.

Video frame encoding runs on a dedicated bounded background queue so the
face/detection worker never blocks on VideoWriter I/O. Segment uploads remain
on a separate executor.
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from slave.logging.logger import get_logger
from slave.processes.minio.video_uploader import VideoUploader
from slave.processes.video.video_recorder import VideoRecorder

log = get_logger(__name__)

_VIDEO_QUEUE_SIZE = 2


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

        self._frame_queue: "queue.Queue[np.ndarray | None]" = queue.Queue(maxsize=_VIDEO_QUEUE_SIZE)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"video-writer-{job_id}",
            daemon=True,
        )
        self._thread.start()

    def feed_frame(self, frame: np.ndarray) -> None:
        """Queue the newest annotated frame without blocking the detection worker."""
        try:
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self._frame_queue.put_nowait(frame)
        except Exception:
            log.exception("failed to queue video frame for job_id=%s", self._job_id)

    def _run(self) -> None:
        while not self._stop_event.is_set() or not self._frame_queue.empty():
            try:
                frame = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if frame is None:
                self._frame_queue.task_done()
                break

            try:
                self._recorder.feed_frame(frame)
            except Exception:
                log.exception("video recording failed for job_id=%s (frame dropped from video)", self._job_id)
            finally:
                self._frame_queue.task_done()

    def _on_segment_ready(self, local_path: str, segment_name: str) -> None:
        self._executor.submit(self._upload_segment, local_path, segment_name)

    def _upload_segment(self, local_path: str, segment_name: str) -> None:
        try:
            self._video_uploader.upload_video_segment(self._job_id, local_path, segment_name)
        except Exception:
            log.exception("video segment upload failed for job_id=%s", self._job_id)

    def close(self) -> None:
        """Drain queued frames, close the recorder, then wait for uploads."""
        self._stop_event.set()
        self._thread.join(timeout=5)

        if self._thread.is_alive():
            log.warning("video writer did not stop within timeout for job_id=%s", self._job_id)

        try:
            self._recorder.close()
        except Exception:
            log.exception("failed to flush final video segment for job_id=%s", self._job_id)

        self._executor.shutdown(wait=True)
