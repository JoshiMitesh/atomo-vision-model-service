"""Composition root for the device health/telemetry/command channel.

Runs entirely independently of the job/detection pipeline
(processes/job/, processes/worker/) — this is about whether the Slave
device itself is alive and reachable for administration, not what it's
detecting. Owns its own NATS connection, separate from the job
subscriber's and the detection publisher's.

Plain core NATS pub/sub (not JetStream) — device identity comes from
config/slave.yaml's `device:` section (config.DeviceConfig):
`device_name` identifies this device in every subject and payload,
replacing the old device_id concept entirely (there is no separate
id, only the name).

Both the health/telemetry subject and the command subject are
per-device (`devices.<device_name>.health` / `.cmd`) — the NATS
broker's ACL for this account only grants publish/subscribe on that
exact pattern, not a flat shared topic (confirmed against the real
broker: a flat "device.health" topic gets a permissions violation).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Optional

import nats

from slave.config import DeviceConfig, NatsConfig
from slave.logging.logger import get_logger
from slave.processes.device.command_subscriber import build_command_handler
from slave.processes.device.registration_publisher import publish_registration
from slave.processes.device.telemetry_publisher import publish_event, telemetry_loop

log = get_logger(__name__)


class DeviceAgent:
    def __init__(
        self,
        nats_config: NatsConfig,
        device_config: DeviceConfig,
        jobs_yaml_path: Optional[str] = None,
    ) -> None:
        self._nats_config = nats_config
        self._device_config = device_config
        self._jobs_yaml_path = jobs_yaml_path
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="device-agent", daemon=True)
        self._stop_event: Optional[asyncio.Event] = None
        self._ready = threading.Event()
        self._nc = None
        self._data_subject: Optional[str] = None

    def start(self) -> None:
        self._thread.start()
        self._ready.wait(timeout=10)
        log.info("device agent started (device_name=%s)", self._device_config.device_name)

    def stop(self) -> None:
        if self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        self._thread.join(timeout=10)
        log.info("device agent stopped")

    def publish_event(self, event: dict) -> None:
        """Publish a one-off device event (e.g. an alarm) — safe to call
        from any thread. No-op (logged) if the agent never finished
        connecting.
        """
        if self._nc is None or self._data_subject is None:
            log.warning("publish_event dropped, device agent not connected: %s", event)
            return
        asyncio.run_coroutine_threadsafe(
            publish_event(self._nc, self._data_subject, event), self._loop
        )

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main())

    async def _connect(self):
        """Connect to config/slave.yaml's configured NATS host —
        retried forever with backoff so a NATS server that's briefly
        down doesn't take the device agent down.
        """
        nc_holder: dict = {}

        async def disconnected_cb():
            log.warning("device agent disconnected from NATS")

        async def reconnected_cb():
            log.info("device agent reconnected to NATS: %s", nc_holder.get("nc") and nc_holder["nc"].connected_url)

        async def error_cb(error):
            log.warning("device agent NATS error: %s", error)

        async def closed_cb():
            log.info("device agent NATS connection closed")

        attempt = 0
        while True:
            host = self._nats_config.host
            scheme = "ws" if self._nats_config.use_websocket else "nats"
            url = f"{scheme}://{host}:{self._nats_config.port}"
            try:
                nc = await nats.connect(
                    servers=[url],
                    user=self._nats_config.user,
                    password=self._nats_config.password,
                    reconnect_time_wait=self._nats_config.reconnect_wait_seconds,
                    max_reconnect_attempts=-1,
                    disconnected_cb=disconnected_cb,
                    reconnected_cb=reconnected_cb,
                    error_cb=error_cb,
                    closed_cb=closed_cb,
                )
                nc_holder["nc"] = nc
                log.info("device agent connected to %s", url)
                return nc
            except Exception as exc:
                attempt += 1
                wait_s = min(self._nats_config.reconnect_wait_seconds * attempt, 30)
                log.warning("device agent could not reach %s (attempt %d): %s — retrying in %.0fs", url, attempt, exc, wait_s)
                await asyncio.sleep(wait_s)

    async def _main(self) -> None:
        self._stop_event = asyncio.Event()
        dc = self._device_config
        nc = None

        try:
            nc = await self._connect()
            self._nc = nc

            device_name = dc.device_name
            nc_cfg = self._nats_config
            data_subject = nc_cfg.health_subject.format(device_name=device_name)  # per-device topic — the NATS broker's ACL for this account only grants publish on this exact pattern, not a flat topic (confirmed: "permissions violation for publish to device.health")
            cmd_subject = nc_cfg.cmd_subject.format(device_name=device_name)
            register_subject = nc_cfg.register_subject.format(device_name=device_name)
            self._data_subject = data_subject

            handler = build_command_handler(device_name, self._jobs_yaml_path, dc.ip)
            await nc.subscribe(cmd_subject, cb=handler)
            log.info("listening for device commands on %s", cmd_subject)

            await publish_registration(nc, register_subject, dc)
        except Exception:
            log.exception("device agent failed to start")
            return
        finally:
            self._ready.set()

        try:
            await telemetry_loop(
                nc, self._data_subject, dc.heartbeat_interval_seconds, self._stop_event, self._jobs_yaml_path,
                dc.device_name, dc.ip,
            )
        finally:
            self._nc = None
            self._data_subject = None
            if nc is not None:
                await nc.drain()
