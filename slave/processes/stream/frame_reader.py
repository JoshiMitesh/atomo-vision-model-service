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
    def __init__(self, job_id: str, url: str, decode_every_n_frames: int = 3) -> None:
        self._job_id = job_id
        self._client = RtspClient(url, decode_every_n_frames=decode_every_n_frames)
        self._queue: "queue.Queue[Frame]" = queue.Queue(maxsize=1)
        self._frame_ids = FrameIdGenerator()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"frame-reader-{job_id}", daemon=True)
        self._log = get_logger(__name__, job_id=job_id)

        self._status_lock = threading.Lock()
        self._capture_fps = 0.0
        self._fps_frame_count = 0
        self._fps_window_start = time.monotonic()
        self._status = "offline"

    def get_status(self) -> str:
        """Camera connection status: 'online', 'reconnecting', or 'offline'."""
        with self._status_lock:
            return self._status
        
    def get_capture_fps(self) -> float:
        """Measured RTSP capture rate over the last ~2s window."""
        with self._status_lock:
            return self._capture_fps

    def _set_status(self, status: str) -> None:
        with self._status_lock:
            self._status = status

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
            error = self._client.get_last_error()
            if error:
                self._log.warning("initial RTSP connect failed, will retry: %s", error)
            else:
                self._log.warning("initial RTSP connect failed, will retry")
            self._set_status("offline")

        while not self._stop_event.is_set():
            ok, image = self._client.read()
            if not ok:
                error = self._client.get_last_error()
                if error:
                    self._log.warning("RTSP read failed, reconnecting: %s", error)
                else:
                    self._log.warning("RTSP read failed, reconnecting")
                self._set_status("reconnecting")
                self._reconnect_with_backoff()
                continue

            if image is None:
                # decode_every_n_frames intentionally skips most frames to
                # reduce CPU work; this is not an RTSP failure. Sleep here
                # so the reader thread does not spin in a tight loop while
                # waiting for the next decodable frame.
                time.sleep(0.01)
                continue

            self._set_status("online")
            frame = Frame(
                job_id=self._job_id,
                frame_id=self._frame_ids.next(),
                image=image,
                timestamp=epoch_seconds(),
            )
            self._push_latest(frame)
            self._tick_fps()

        self._set_status("offline")
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

    def _tick_fps(self) -> None:
        with self._status_lock:
            self._fps_frame_count += 1
            elapsed = time.monotonic() - self._fps_window_start
            if elapsed >= 2.0:
                self._capture_fps = self._fps_frame_count / elapsed
                self._fps_frame_count = 0
                self._fps_window_start = time.monotonic()


    def _reconnect_with_backoff(self) -> None:
        for _ in range(3):
            if self._stop_event.is_set():
                return
            time.sleep(RECONNECT_BACKOFF_SECONDS)
            if self._client.reconnect():
                self._log.info("RTSP reconnected")
                return
        self._set_status("offline")
        self._log.error("RTSP reconnect failing repeatedly, will keep retrying")
