"""One 30-second local video segment file, backed by cv2.VideoWriter."""

from __future__ import annotations

import os
import time

import cv2
import numpy as np


class VideoSegment:
    def __init__(self, path: str, fps: int, frame_size: tuple[int, int]) -> None:
        self.path = path
        self._fps = fps
        self._frame_size = frame_size
        self._writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
        self._started_at = time.monotonic()

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self._started_at

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1::-1] != self._frame_size:
            frame = cv2.resize(frame, self._frame_size)
        self._writer.write(frame)

    def close(self) -> str:
        self._writer.release()
        return self.path

    def discard(self) -> None:
        self._writer.release()
        if os.path.exists(self.path):
            os.remove(self.path)
