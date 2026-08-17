"""NATS subscriber — the only entry point for job/stop control messages.

Responsibilities: receive, validate, persist. Nothing else.
It must NEVER instantiate PersonPtWorker/PersonNbWorker/PersonSoWorker
directly (CLAUDE.md §5) — job_watcher.py is solely responsible for
turning jobs.yaml changes into running workers.
"""

from __future__ import annotations

import asyncio
import json

import nats

from slave.config import NatsConfig
from slave.logging.logger import get_logger
from slave.models.message import MessageType, classify
from slave.processes.job.stop_handler import handle_stop
from slave.processes.subscriber.job_message import InvalidJobMessage, parse_job_message
from slave.processes.subscriber.stop_message import InvalidStopMessage, parse_stop_message
from slave.storage.yaml import job_store

log = get_logger(__name__)


class NatsSubscriber:
    """Async NATS subscriber for job-control messages.

    Runs its own asyncio event loop (see main.py) so the rest of the
    Slave, which is thread-based, can stay synchronous.
    """

    def __init__(self, nats_config: NatsConfig, jobs_yaml_path: str) -> None:
        self._config = nats_config
        self._jobs_yaml_path = jobs_yaml_path
        self._nc: nats.aio.client.Client | None = None

    async def start(self) -> None:
        self._nc = await nats.connect(
            self._config.url,
            user=self._config.user,
            password=self._config.password,
            reconnect_time_wait=self._config.reconnect_wait_seconds,
            max_reconnect_attempts=-1,
        )
        await self._nc.subscribe(self._config.job_subject, cb=self._on_message)
        await self._nc.subscribe(self._config.stop_subject, cb=self._on_message)
        log.info(
            "NATS subscriber connected to %s (job=%s, stop=%s)",
            self._config.url,
            self._config.job_subject,
            self._config.stop_subject,
        )

    async def stop(self) -> None:
        if self._nc is not None:
            await self._nc.drain()
            self._nc = None
            log.info("NATS subscriber disconnected")

    async def _on_message(self, msg) -> None:
        try:
            raw = json.loads(msg.data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            log.warning("dropped non-JSON NATS message on %s", msg.subject)
            return

        message_type = classify(raw)
        try:
            if message_type is MessageType.JOB:
                self._handle_job(raw)
            elif message_type is MessageType.STOP:
                self._handle_stop(raw)
            else:
                log.warning("dropped unrecognized NATS message on %s: %s", msg.subject, raw)
        except Exception:
            # A single malformed/unexpected message must never crash the
            # subscriber or affect other jobs.
            log.exception("error handling NATS message on %s", msg.subject)

    def _handle_job(self, raw: dict) -> None:
        try:
            job = parse_job_message(raw)
        except InvalidJobMessage as exc:
            log.warning("rejected invalid job message: %s", exc)
            return
        job_store.upsert_job(self._jobs_yaml_path, job)
        log.info("job message stored (job_id=%s, model_type=%s)", job.job_id, job.model_type)

    def _handle_stop(self, raw: dict) -> None:
        try:
            job_id = parse_stop_message(raw)
        except InvalidStopMessage as exc:
            log.warning("rejected invalid stop message: %s", exc)
            return
        handle_stop(self._jobs_yaml_path, job_id)


def run_subscriber_forever(nats_config: NatsConfig, jobs_yaml_path: str, stop_event: asyncio.Event) -> None:
    """Blocking entry point intended to run inside its own thread via
    asyncio.run — see main.py.
    """

    async def _main() -> None:
        subscriber = NatsSubscriber(nats_config, jobs_yaml_path)
        await subscriber.start()
        try:
            await stop_event.wait()
        finally:
            await subscriber.stop()

    asyncio.run(_main())
