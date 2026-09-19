"""Generic thread-safe NATS publisher.

Worker threads are synchronous; NATS is asyncio-native. This class
runs one background event loop and exposes a plain, blocking-free
``publish`` method that any worker thread can call safely.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import nats

from slave.config import NatsConfig
from slave.logging.logger import get_logger

log = get_logger(__name__)


# calls in main.py:
class Publisher:
    def __init__(self, nats_config: NatsConfig) -> None:
        self._config = nats_config # takes the values from config.py
        self._loop = asyncio.new_event_loop() # runs the NATS client
        self._nc: nats.aio.client.Client | None = None # NATS client instance
        self._thread = threading.Thread(target=self._run_loop, name="publisher-loop", daemon=True) # background thread for the event loop
        self._ready = threading.Event() # event to signal when the publisher is ready to publish messages
        self._last_drop_log = 0.0  # monotonic timestamp of the last "dropping message" warning — rate-limits that log while disconnected instead of one line per dropped message

    def _run_loop(self) -> None: # runs the event loop in a separate thread
        asyncio.set_event_loop(self._loop) # sets the event loop for the current thread to the newly created event loop
        self._loop.create_task(self._connect_with_retry()) # kicks off the initial connect in the background — never blocks this loop from starting
        self._loop.run_forever() # runs the event loop indefinitely, allowing the publisher to handle incoming messages and publish messages to the NATS server

    async def _connect_with_retry(self) -> None:
        """Keep trying to reach NATS forever, without blocking job/worker
        startup. The master (or NATS server) may not be reachable when a
        job starts — that's fine, detection still runs; publish() just
        no-ops (drops) until this succeeds. Once connected, whenever the
        master/NATS comes up later, this picks it up automatically and
        starts publishing — no restart needed."""
        async def error_cb(error):
            # Without this, async server-sent errors (e.g. a permissions
            # violation on publish) are invisible — nats.py has nowhere
            # else to report them since publish() is fire-and-forget.
            log.warning("publisher NATS error: %s", error)

        attempt = 0
        while True:
            try:
                self._nc = await nats.connect(
                    self._config.url, # URL of the NATS server takes from slave.yaml
                    user=self._config.user, # username for authentication takes from slave.yaml
                    password=self._config.password, # password for authentication takes from slave.yaml
                    reconnect_time_wait=self._config.reconnect_wait_seconds, # time to wait before attempting to reconnect takes from slave.yaml
                    max_reconnect_attempts=-1, # unlimited reconnect attempts (after this first successful connect)
                    error_cb=error_cb,
                )
                log.info("publisher connected to %s", self._config.url) # logs a message indicating that the publisher has successfully connected to the NATS server
                self._ready.set() # signals that the publisher is ready to publish messages
                return
            except Exception as exc:
                attempt += 1
                wait_s = min(self._config.reconnect_wait_seconds * attempt, 30) # gentle backoff, capped at 30s
                log.warning(
                    "publisher could not reach %s (attempt %d): %s — retrying in %.0fs",
                    self._config.url, attempt, exc, wait_s,
                )
                await asyncio.sleep(wait_s)

    def start(self) -> None: # starts the publisher by starting the background thread that runs the event loop; does NOT wait for NATS to actually be reachable
        self._thread.start() # starts the background thread that runs the event loop
        self._ready.wait(timeout=2) # brief grace period only, in case NATS happens to already be up — never blocks job startup on an unreachable master

    def publish(self, subject: str, payload: dict) -> None: # publishes a message to the specified subject on the NATS server, using the provided payload. This method is thread-safe and can be called from any worker thread.
        """Fire-and-forget publish, safe to call from any worker thread."""
        if self._nc is None:
            # While disconnected, a worker running at full frame rate can
            # call this dozens of times a second — logging every single
            # drop floods the terminal with near-duplicate lines. Log a
            # sample at most once every 5s instead; every other drop in
            # between is silent (the messages are gone either way, this
            # only controls how noisy that fact is).
            now = time.monotonic()
            if now - self._last_drop_log >= 5.0:
                self._last_drop_log = now
                log.warning("NATS not connected — dropping messages on %s (sample payload=%s)", subject, json.dumps(payload))
            return
        data = json.dumps(payload).encode("utf-8") # serializes the payload to a JSON string and encodes it as bytes
        asyncio.run_coroutine_threadsafe(self._nc.publish(subject, data), self._loop)  # schedules the publish coroutine to be run on the event loop in the background thread, allowing the message to be published to the NATS server without blocking the calling thread

    def stop(self) -> None: # stops the publisher by draining the NATS client, stopping the event loop, and joining the background thread. This method is thread-safe and can be called from any worker thread.
        """Stop the publisher, waiting for any pending messages to be sent."""
        if self._nc is not None:
            asyncio.run_coroutine_threadsafe(self._nc.drain(), self._loop).result(timeout=5) # drains the NATS client, waiting for any pending messages to be sent before closing the connection
        self._loop.call_soon_threadsafe(self._loop.stop) # stops the event loop in the background thread, allowing the publisher to clean up any resources and exit gracefully
        self._thread.join(timeout=5)   # waits for the background thread to finish executing, with a timeout of 5 seconds. If the thread does not finish within the timeout, the method will return and the publisher may not have fully cleaned up its resources.
        log.info("publisher stopped") # logs a message indicating that the publisher has successfully stopped and cleaned up its resources
