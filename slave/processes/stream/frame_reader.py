"""Background thread that keeps only the newest frame available.

Bounded queue(maxsize=1), drop-stale/insert-newest strategy
(CLAUDE.md §15), automatic reconnect, clean shutdown. One instance per
job — never shared (CLAUDE.md §20).
"""

from __future__ import annotations

import queue
import threading
import time

from slave.logging.logger import get_logger
from slave.models.frame import Frame
from slave.processes.stream.rtsp_client import RtspClient
from slave.utils.id_generator import FrameIdGenerator
from slave.utils.time_utils import epoch_seconds

RECONNECT_BACKOFF_SECONDS = 2.0


class FrameReader:
    def __init__(self, job_id: str, url: str) -> None:
        self._job_id = job_id
        self._client = RtspClient(url)
        self._queue: "queue.Queue[Frame]" = queue.Queue(maxsize=1)
        self._frame_ids = FrameIdGenerator()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"frame-reader-{job_id}", daemon=True)
        self._log = get_logger(__name__, job_id=job_id)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5)
        self._client.release()

    def get_latest_frame(self, timeout: float = 1.0) -> Frame | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _run(self) -> None:
        if not self._client.connect():
            self._log.warning("initial RTSP connect failed, will retry")

        while not self._stop_event.is_set():
            ok, image = self._client.read()
            if not ok or image is None:
                self._log.warning("RTSP read failed, reconnecting")
                self._reconnect_with_backoff()
                continue

            frame = Frame(
                job_id=self._job_id,
                frame_id=self._frame_ids.next(),
                image=image,
                timestamp=epoch_seconds(),
            )
            self._push_latest(frame)

        self._log.info("frame reader stopped")

    def _push_latest(self, frame: Frame) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()  # drop stale frame
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            pass

    def _reconnect_with_backoff(self) -> None:
        for _ in range(3):
            if self._stop_event.is_set():
                return
            time.sleep(RECONNECT_BACKOFF_SECONDS)
            if self._client.reconnect():
                self._log.info("RTSP reconnected")
                return
        self._log.error("RTSP reconnect failing repeatedly, will keep retrying")
