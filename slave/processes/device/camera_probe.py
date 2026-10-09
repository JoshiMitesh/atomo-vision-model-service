"""Independent camera reachability check for the device health message.

A camera's health status must not depend on whether a model happens to be
running on it: a job whose model failed to build/start has no worker (so no
RTSP reader) but its camera can be perfectly fine. This opens the job's
stream on its own, briefly, in a background thread, and caches the result
so the telemetry loop never blocks on the network.

Only used when no worker already reports the camera online -- a running
worker's own live connection is the better source and costs nothing extra.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Tuple

import cv2

from slave.logging.logger import get_logger

log = get_logger(__name__)

PROBE_TTL_SECONDS = 15.0
PROBE_TIMEOUT_MS = 4000


def probe_stream(url: str, timeout_ms: int = PROBE_TIMEOUT_MS) -> bool:
    """True if the stream opens and delivers a frame within the timeout."""
    cap = None
    try:
        cap = cv2.VideoCapture(
            url,
            cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms],
        )
        return bool(cap.isOpened() and cap.grab())
    except Exception:
        return False
    finally:
        if cap is not None:
            cap.release()


class CameraProbe:
    def __init__(self, ttl_seconds: float = PROBE_TTL_SECONDS, timeout_ms: int = PROBE_TIMEOUT_MS) -> None:
        self._ttl = ttl_seconds
        self._timeout_ms = timeout_ms
        self._lock = threading.Lock()
        self._results: Dict[Tuple[str, str], Tuple[bool, float]] = {}  # (job_id, url) -> (online, checked_at)
        self._in_flight: set = set()

    def status(self, job_id: str, url: str) -> str:
        """'online' / 'offline' from the last probe. Never blocks: a stale
        or missing result triggers a background refresh and the previous
        answer (offline until the first probe finishes) is returned now.
        """
        key = (job_id, url)
        with self._lock:
            result = self._results.get(key)
            stale = result is None or time.monotonic() - result[1] >= self._ttl
            start = stale and key not in self._in_flight
            if start:
                self._in_flight.add(key)
        if start:
            threading.Thread(target=self._refresh, args=(key,), name=f"camera-probe-{job_id}", daemon=True).start()
        return "online" if result is not None and result[0] else "offline"

    def forget(self, job_id: str) -> None:
        with self._lock:
            for key in [k for k in self._results if k[0] == job_id]:
                del self._results[key]

    def _refresh(self, key: Tuple[str, str]) -> None:
        try:
            online = probe_stream(key[1], self._timeout_ms)
        except Exception:
            log.exception("camera probe failed for job_id=%s", key[0])
            online = False
        with self._lock:
            self._results[key] = (online, time.monotonic())
            self._in_flight.discard(key)
