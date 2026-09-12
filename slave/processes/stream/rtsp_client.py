"""Low-latency OpenCV/FFmpeg RTSP capture (CLAUDE.md §14).

Configures TCP transport, no buffering, and a 1-frame internal buffer
so the newest frame is always what gets read.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np

_FFMPEG_OPTIONS = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0|buffer_size;65536|threads;1"


class RtspClient:
    def __init__(self, url: str, decode_every_n_frames: int = 8) -> None:
        self._url = url
        self._cap: cv2.VideoCapture | None = None
        self._decode_every_n_frames = max(1, int(decode_every_n_frames))
        self._read_counter = 0

    def connect(self) -> bool:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _FFMPEG_OPTIONS
        self._cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if hasattr(cv2, "CAP_PROP_THREAD_COUNT"):
            # Belt-and-suspenders alongside threads;1 above and the
            # process-wide cv2.setNumThreads(1) in main.py.
            self._cap.set(cv2.CAP_PROP_THREAD_COUNT, 1)
        return self._cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        # grab() is lighter than retrieve() and lets us skip expensive
        # decode work for stale frames. This matches the reference script's
        # low-CPU pattern: only decode every Nth frame instead of decoding
        # every RTSP packet the camera sends.
        if not self._cap.grab():
            return False, None
        self._read_counter += 1
        if self._read_counter % self._decode_every_n_frames != 0:
            return True, None
        return self._cap.retrieve()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def reconnect(self) -> bool:
        self.release()
        return self.connect()
