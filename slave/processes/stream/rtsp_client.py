"""Low-latency OpenCV/FFmpeg RTSP capture (CLAUDE.md §14).

Configures TCP transport, no buffering, and a 1-frame internal buffer
so the newest frame is always what gets read.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np

_FFMPEG_OPTIONS = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0|buffer_size;65536"


class RtspClient:
    def __init__(self, url: str) -> None:
        self._url = url
        self._cap: cv2.VideoCapture | None = None

    def connect(self) -> bool:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _FFMPEG_OPTIONS
        self._cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self._cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        # grab()+retrieve() discards any stale buffered frame first.
        if not self._cap.grab():
            return False, None
        return self._cap.retrieve()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def reconnect(self) -> bool:
        self.release()
        return self.connect()
