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
    def __init__(self, publisher: Publisher, upload_subject: str) -> None: # initializes the UploadPublisher with a NATS publisher and upload subject
        self._publisher = publisher # publisher is an instance of the Publisher class, which handles the actual NATS publishing
        self._subject = upload_subject # subject is the NATS subject to which upload events will be published

    def publish_upload_started( 
        self,
        job_id: str,
        kind: str,
        object_key: str,
        url: str,
        frame_id: Optional[int] = None,
        segment_name: Optional[str] = None,
    ) -> None: # publishes an upload-started event to the specified subject on the NATS server, using the provided upload details. This method is thread-safe and can be called from any worker thread.
        message = build_upload_message(job_id, kind, object_key, url, frame_id, segment_name) # build_upload_message is a function that takes the upload details and converts them into a message format suitable for publishing over NATS
        try:
            self._publisher.publish(self._subject, message) # publish is a method of the Publisher class that sends the message to the specified NATS subject
        except Exception:
            log.exception(
                "failed to publish upload-started event for job_id=%s kind=%s key=%s",
                job_id, kind, object_key,
            ) # log the exception if publishing fails
