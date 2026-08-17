"""Composition root for the device health/telemetry/command channel.

Runs entirely independently of the job/detection pipeline
(processes/job/, processes/worker/) — this is about whether the Slave
device itself is alive and reachable for administration, not what it's
detecting. Owns its own NATS connection, separate from the job
subscriber's and the detection publisher's.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Optional

import nats

from slave.config import DeviceConfig, NatsConfig
from slave.logging.logger import get_logger
from slave.processes.device.command_subscriber import build_command_handler
from slave.processes.device.heartbeat_publisher import heartbeat_loop
from slave.processes.device.telemetry_publisher import ensure_data_stream, publish_event, telemetry_loop

log = get_logger(__name__)


class DeviceAgent:
    def __init__(self, nats_config: NatsConfig, device_config: DeviceConfig) -> None:
        self._nats_config = nats_config
        self._device_config = device_config
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="device-agent", daemon=True)
        self._stop_event: Optional[asyncio.Event] = None
        self._ready = threading.Event()
        self._js = None
        self._data_subject: Optional[str] = None

    def start(self) -> None:
        self._thread.start()
        self._ready.wait(timeout=10)
        log.info(
            "device agent started (site_id=%s, slave_id=%s)",
            self._device_config.site_id, self._device_config.slave_id,
        )

    def stop(self) -> None:
        if self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        self._thread.join(timeout=10)
        log.info("device agent stopped")

    def publish_event(self, event: dict) -> None:
        """Publish a one-off device event (e.g. an alarm) — safe to call
        from any thread. No-op (logged) if the agent never finished
        connecting (e.g. JetStream unavailable at startup).
        """
        if self._js is None or self._data_subject is None:
            log.warning("publish_event dropped, device agent not connected: %s", event)
            return
        asyncio.run_coroutine_threadsafe(
            publish_event(self._js, self._data_subject, event), self._loop
        )

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main())

    async def _main(self) -> None:
        self._stop_event = asyncio.Event()
        dc = self._device_config
        nc = None

        try:
            nc = await nats.connect(
                self._nats_config.url,
                user=self._nats_config.user,
                password=self._nats_config.password,
                reconnect_time_wait=self._nats_config.reconnect_wait_seconds,
                max_reconnect_attempts=-1,
            )
            js = nc.jetstream()
            self._js = js

            data_subject = f"site.{dc.site_id}.device.{dc.slave_id}.data"
            cmd_subject = f"site.{dc.site_id}.device.{dc.slave_id}.cmd"
            self._data_subject = data_subject

            await ensure_data_stream(js, dc.data_stream_name, ["site.*.device.*.data"])

            handler = build_command_handler(dc.slave_id)
            await nc.subscribe(cmd_subject, cb=handler)
            log.info("listening for device commands on %s", cmd_subject)
        except Exception:
            log.exception("device agent failed to start")
            return
        finally:
            self._ready.set()

        try:
            await asyncio.gather(
                telemetry_loop(js, data_subject, dc.data_interval_seconds, self._stop_event),
                heartbeat_loop(
                    js, dc.health_bucket, dc.health_ttl_seconds,
                    f"{dc.site_id}.{dc.slave_id}", dc.heartbeat_interval_seconds, self._stop_event,
                ),
            )
        finally:
            self._js = None
            self._data_subject = None
            if nc is not None:
                await nc.drain()
