"""Publishes normalized detection results to the Master over NATS."""

from __future__ import annotations

from slave.logging.logger import get_logger
from slave.processes.detection.detection_message import build_detection_message
from slave.processes.detection.detection_result import DetectionResult
from slave.processes.publisher.publisher import Publisher

log = get_logger(__name__)


class DetectionPublisher:
    def __init__(self, publisher: Publisher, detection_subject: str) -> None: # 
        self._publisher = publisher # publisher is an instance of the Publisher class, which handles the actual NATS publishing
        self._subject = detection_subject # subject is the NATS subject to which detections will be published

    def publish_detection(self, result: DetectionResult) -> None:
        # Frames with nothing detected are never sent to the Master.
        if not result.has_detections:
            return
        message = build_detection_message(result) # build_detection_message is a function that takes a DetectionResult and converts it into a message format suitable for publishing over NATS
        try:
            self._publisher.publish(self._subject, message) # publish is a method of the Publisher class that sends the message to the specified NATS subject
        except Exception:
            log.exception("failed to publish detection for job_id=%s frame_id=%s", result.job_id, result.frame_id) # log the exception if publishing fails
