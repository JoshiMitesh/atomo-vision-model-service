"""Generic thread-safe NATS publisher.

Worker threads are synchronous; NATS is asyncio-native. This class
runs one background event loop and exposes a plain, blocking-free
``publish`` method that any worker thread can call safely.
"""

from __future__ import annotations

import asyncio
import json
import threading

import nats

from slave.config import NatsConfig
from slave.logging.logger import get_logger

log = get_logger(__name__)


class Publisher:
    def __init__(self, nats_config: NatsConfig) -> None:
        self._config = nats_config
        self._loop = asyncio.new_event_loop()
        self._nc: nats.aio.client.Client | None = None
        self._thread = threading.Thread(target=self._run_loop, name="publisher-loop", daemon=True)
        self._ready = threading.Event()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect())
        self._ready.set()
        self._loop.run_forever()

    async def _connect(self) -> None:
        self._nc = await nats.connect(
            self._config.url,
            user=self._config.user,
            password=self._config.password,
            reconnect_time_wait=self._config.reconnect_wait_seconds,
            max_reconnect_attempts=-1,
        )
        log.info("publisher connected to %s", self._config.url)

    def start(self) -> None:
        self._thread.start()
        self._ready.wait(timeout=10)

    def publish(self, subject: str, payload: dict) -> None:
        """Fire-and-forget publish, safe to call from any worker thread."""
        if self._nc is None:
            log.warning("publish attempted before connection ready, dropping message on %s", subject)
            return
        data = json.dumps(payload).encode("utf-8")
        asyncio.run_coroutine_threadsafe(self._nc.publish(subject, data), self._loop)

    def stop(self) -> None:
        if self._nc is not None:
            asyncio.run_coroutine_threadsafe(self._nc.drain(), self._loop).result(timeout=5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        log.info("publisher stopped")
