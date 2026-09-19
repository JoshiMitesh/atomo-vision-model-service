"""Low-latency OpenCV/FFmpeg RTSP capture (CLAUDE.md §14).

Configures TCP transport, no buffering, and a 1-frame internal buffer
so the newest frame is always what gets read.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np

_FFMPEG_OPTIONS = (
    "rtsp_transport;tcp|"
    "fflags;nobuffer+discardcorrupt|"
    "flags;low_delay|"
    "max_delay;0|"
    "buffer_size;65536|"
    "analyzeduration;1000000|"
    "probesize;1000000|"
    "reorder_queue_size;0|"
    "threads;1"
)


class RtspClient:
    def __init__(self, url: str, decode_every_n_frames: int = 1) -> None:
        self._url = url
        self._cap: cv2.VideoCapture | None = None
        # >1 skips retrieve() (the expensive decode) on most grab()
        # calls — grab() itself stays cheap and unconditional so the
        # RTSP buffer never backs up. Default 1 = decode every frame,
        # unchanged from prior behavior; callers opt into skipping.
        self._decode_every_n_frames = max(1, int(decode_every_n_frames))
        self._read_counter = 0
        self._last_error = ""

    def connect(self) -> bool:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _FFMPEG_OPTIONS
        self._cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if hasattr(cv2, "CAP_PROP_THREAD_COUNT"):
            # Belt-and-suspenders alongside threads;1 above and the
            # process-wide cv2.setNumThreads(1) in main.py.
            self._cap.set(cv2.CAP_PROP_THREAD_COUNT, 1)
        ok = self._cap.isOpened()
        if not ok:
            self._last_error = "OpenCV could not open RTSP stream"
        return ok

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            self._last_error = "video capture is not initialized"
            return False, None

        self._read_counter += 1
        if self._read_counter % self._decode_every_n_frames != 0:
            return True, None

        # grab()+retrieve() discards any stale buffered frame first.
        if not self._cap.grab():
            self._last_error = "OpenCV grab() failed"
            return False, None

        ok, frame = self._cap.retrieve()
        if not ok or frame is None:
            # Corrupt H.264 packets can make FFmpeg fail a single frame
            # without the entire RTSP stream being dead. Treat that as
            # a dropped frame so we do not reconnect the whole stream
            # on every bad packet. Real connection loss still triggers
            # the grab() failure path above.
            self._last_error = "OpenCV retrieve() failed; skipping corrupt frame"
            return True, None

        return True, frame

    def get_last_error(self) -> str:
        return self._last_error

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def reconnect(self) -> bool:
        self.release()
        return self.connect()
