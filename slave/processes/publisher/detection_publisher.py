"""Publishes normalized detection results to the Master over NATS."""

from __future__ import annotations

from slave.logging.logger import get_logger
from slave.processes.detection.detection_message import build_detection_message
from slave.processes.detection.detection_result import DetectionResult
from slave.processes.publisher.publisher import Publisher

log = get_logger(__name__)


class DetectionPublisher:
    def __init__(self, publisher: Publisher, detection_subject: str) -> None:
        self._publisher = publisher
        self._subject = detection_subject

    def publish_detection(self, result: DetectionResult) -> None:
        message = build_detection_message(result)
        try:
            self._publisher.publish(self._subject, message)
        except Exception:
            log.exception("failed to publish detection for job_id=%s frame_id=%s", result.job_id, result.frame_id)
