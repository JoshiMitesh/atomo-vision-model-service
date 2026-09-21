"""Publishes periodic device telemetry to the Master over core NATS.

Plain publish/subscribe (not JetStream) — this device's NATS server is
config/slave.yaml's configured nats.host, and the same connection
retry loop in device_agent.py already tolerates it being briefly
unreachable. A dropped telemetry tick just means the Master sees the
next one a few seconds later; nothing here needs at-least-once
delivery.
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from slave.logging.logger import get_logger
from slave.processes.device.telemetry import sample_telemetry

log = get_logger(__name__)


async def publish_event(nc, subject: str, event: dict) -> None:
    """Publish a one-off, non-periodic device event (e.g. an alarm) on
    the same subject the periodic telemetry loop uses. Distinct from
    processes/publisher/detection_publisher.py — that's for per-job
    detection results, this is for device-level events.

    Errors are caught and logged here (not left to the caller) because
    DeviceAgent.publish_event() fires this via
    run_coroutine_threadsafe() without inspecting the result — an
    unhandled exception here would otherwise be silently dropped.
    """
    try:
        await nc.publish(subject, json.dumps(event).encode())
        log.info("event published on %s: %s", subject, event)
    except Exception:
        log.exception("failed to publish event on %s: %s", subject, event)


async def telemetry_loop(
    nc,
    subject: str,
    interval_seconds: float,
    stop_event: asyncio.Event,
    jobs_yaml_path: Optional[str],
    device_name: str,
    ip: str,
) -> None:
    """Never raises out of this loop — a publish failure here must not
    take down the device agent's other loop (the command subscription)
    or its NATS connection, so every failure is caught and retried on
    the next tick instead.
    """
    log.info("publishing telemetry on %s every %ss", subject, interval_seconds)
    while not stop_event.is_set():
        try:
            if nc.is_connected:
                payload = sample_telemetry(jobs_yaml_path, device_name, ip)
                await nc.publish(subject, json.dumps(payload).encode())
            else:
                log.warning("device NATS connection down, skipping telemetry tick")
        except Exception as exc:
            log.warning("telemetry publish failed (will retry): %s", exc)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            pass
