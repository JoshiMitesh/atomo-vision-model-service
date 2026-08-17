"""Publishes periodic presence heartbeats into a TTL'd KV bucket.

A missed heartbeat auto-expires from the bucket, so the Master detects
this device going offline just by the key's absence — no explicit
disconnect message needed.
"""

from __future__ import annotations

import asyncio
import json

from slave.logging.logger import get_logger
from slave.processes.device.telemetry import sample_telemetry

log = get_logger(__name__)


async def _get_or_create_kv(js, bucket_name: str, ttl_seconds: int):
    try:
        return await js.key_value(bucket_name)
    except Exception:
        return await js.create_key_value(bucket=bucket_name, ttl=ttl_seconds)


async def heartbeat_loop(
    js,
    bucket_name: str,
    ttl_seconds: int,
    key: str,
    interval_seconds: float,
    stop_event: asyncio.Event,
) -> None:
    """Never raises out of this loop — a JetStream/KV problem here must
    not take down the device agent's other loops or its NATS connection
    (e.g. the command-handler subscription), so every failure is caught
    and retried on the next tick instead.
    """
    kv = None
    log.info(
        "heartbeating to bucket '%s' key '%s' every %ss (ttl=%ss)",
        bucket_name, key, interval_seconds, ttl_seconds,
    )

    while not stop_event.is_set():
        try:
            if kv is None:
                kv = await _get_or_create_kv(js, bucket_name, ttl_seconds)
            payload = sample_telemetry()
            await kv.put(key, json.dumps(payload).encode())
        except Exception as exc:
            log.warning("heartbeat failed (will retry): %s", exc)
            kv = None  # re-acquire next tick in case the bucket/connection recovers
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            pass
