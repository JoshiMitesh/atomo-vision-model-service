"""Publishes MinIO upload-started events to the Master over NATS.

Kept separate from DetectionPublisher (CLAUDE.md §17/§22: don't put
MinIO logic inside detectors, don't mix responsibilities) — this only
knows how to announce "an upload started, here's where it will live",
not how to perform the upload itself.
"""

from __future__ import annotations

from typing import Optional

from slave.logging.logger import get_logger
from slave.processes.publisher.publisher import Publisher
from slave.processes.publisher.upload_message import build_upload_message

log = get_logger(__name__)


class UploadPublisher:
    def __init__(self, publisher: Publisher, upload_subject: str) -> None:
        self._publisher = publisher
        self._subject = upload_subject

    def publish_upload_started(
        self,
        job_id: str,
        kind: str,
        object_key: str,
        url: str,
        frame_id: Optional[int] = None,
        segment_name: Optional[str] = None,
    ) -> None:
        message = build_upload_message(job_id, kind, object_key, url, frame_id, segment_name)
        try:
            self._publisher.publish(self._subject, message)
        except Exception:
            log.exception(
                "failed to publish upload-started event for job_id=%s kind=%s key=%s",
                job_id, kind, object_key,
            )
