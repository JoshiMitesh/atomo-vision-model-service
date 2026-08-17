"""Rolls incoming frames into consecutive 30-second local segments."""

from __future__ import annotations

import os
import threading
from typing import Callable, Optional

import numpy as np

from slave.logging.logger import get_logger
from slave.processes.video.video_segment import VideoSegment
from slave.utils.file_utils import ensure_dir

log = get_logger(__name__)

SegmentReadyCallback = Callable[[str, str], None]  # (local_path, segment_name) -> None


class VideoRecorder:
    def __init__(
        self,
        job_id: str,
        temp_dir: str,
        segment_seconds: int,
        fps: int,
        on_segment_ready: SegmentReadyCallback,
    ) -> None:
        self._job_id = job_id
        self._temp_dir = ensure_dir(os.path.join(temp_dir, job_id))
        self._segment_seconds = segment_seconds
        self._fps = fps
        self._on_segment_ready = on_segment_ready
        self._segment: Optional[VideoSegment] = None
        self._segment_index = 0
        self._lock = threading.Lock()

    def feed_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            height, width = frame.shape[:2]
            if self._segment is None:
                self._open_new_segment((width, height))
            elif self._segment.age_seconds >= self._segment_seconds:
                self._close_current(upload=True)
                self._open_new_segment((width, height))
            try:
                self._segment.write(frame)
            except Exception:
                log.exception("failed writing frame to video segment for job_id=%s", self._job_id)

    def _open_new_segment(self, frame_size: tuple[int, int]) -> None:
        self._segment_index += 1
        name = f"segment_{self._segment_index:04d}.mp4"
        path = os.path.join(self._temp_dir, name)
        self._segment = VideoSegment(path, self._fps, frame_size)
        self._segment_name = name

    def _close_current(self, upload: bool) -> None:
        if self._segment is None:
            return
        path = self._segment.close()
        name = self._segment_name
        self._segment = None
        if upload:
            self._on_segment_ready(path, name)

    def close(self) -> None:
        """Flush whatever is currently recording (job shutdown, CLAUDE.md §18)."""
        with self._lock:
            self._close_current(upload=True)
