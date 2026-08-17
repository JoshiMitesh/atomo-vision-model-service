"""Publishes periodic device telemetry to the Master over JetStream.

JetStream (not core NATS pub/sub) so telemetry persists even if the
Master is briefly offline — it catches up once reconnected.
"""

from __future__ import annotations

import asyncio
import json
from typing import List

from slave.logging.logger import get_logger
from slave.processes.device.telemetry import sample_telemetry

log = get_logger(__name__)


async def ensure_data_stream(js, stream_name: str, subjects: List[str]) -> None:
    """Idempotent — creates the stream if missing, no-ops otherwise."""
    from nats.js.api import RetentionPolicy, StorageType, StreamConfig

    try:
        await js.stream_info(stream_name)
    except Exception:
        try:
            await js.add_stream(
                StreamConfig(
                    name=stream_name,
                    subjects=subjects,
                    retention=RetentionPolicy.LIMITS,
                    storage=StorageType.FILE,
                    num_replicas=1,
                )
            )
            log.info("created JetStream stream '%s'", stream_name)
        except Exception as exc:
            # Another Slave (or the Master) likely created it concurrently.
            log.debug("stream ensure skipped for '%s': %s", stream_name, exc)


async def publish_event(js, subject: str, event: dict) -> None:
    """Publish a one-off, non-periodic device event (e.g. an alarm) on
    the same data subject the periodic telemetry loop uses. Distinct
    from processes/publisher/detection_publisher.py — that's for
    per-job detection results, this is for device-level events.

    Errors are caught and logged here (not left to the caller) because
    DeviceAgent.publish_event() fires this via
    run_coroutine_threadsafe() without inspecting the result — an
    unhandled exception here would otherwise be silently dropped.
    """
    try:
        await js.publish(subject, json.dumps(event).encode())
        log.info("event published on %s: %s", subject, event)
    except Exception:
        log.exception("failed to publish event on %s: %s", subject, event)


async def telemetry_loop(js, subject: str, interval_seconds: float, stop_event: asyncio.Event) -> None:
    log.info("publishing telemetry on %s every %ss", subject, interval_seconds)
    while not stop_event.is_set():
        payload = sample_telemetry()
        try:
            await js.publish(subject, json.dumps(payload).encode())
        except Exception as exc:
            log.warning("telemetry publish failed: %s", exc)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            pass
