"""Publisher for model profiling telemetry.

Publishes telemetry dictionaries produced by :class:`slave.profiling.profiler.Profiler` to the
configured NATS profiling subject.
"""

from __future__ import annotations

from slave.logging.logger import get_logger
from slave.processes.publisher.publisher import Publisher

log = get_logger(__name__)


class ProfilingPublisher:
    def __init__(self, publisher: Publisher, profiling_subject: str) -> None:
        self._publisher = publisher
        self._subject = profiling_subject

    def publish(self, telemetry: dict) -> None:
        """Publish a profiling telemetry payload.

        The payload is expected to be a JSON‑serializable ``dict``.
        """
        try:
            self._publisher.publish(self._subject, telemetry)
        except Exception:
            log.exception("failed to publish profiling telemetry")
