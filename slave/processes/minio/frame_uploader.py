"""Uploads detected frames to MinIO.

This lives outside the detectors on purpose (CLAUDE.md §17, §22) —
detectors only produce Detection objects, workers decide whether/what
to upload.
"""

from __future__ import annotations

import io

import cv2
import numpy as np

from slave.logging.logger import get_logger
from slave.processes.minio.bucket_manager import frame_object_key
from slave.processes.minio.minio_client import MinioClientWrapper

log = get_logger(__name__)


class FrameUploader:
    def __init__(self, minio_client: MinioClientWrapper) -> None:
        self._minio = minio_client

    def upload_frame(self, job_id: str, frame_id: int, image: np.ndarray) -> None:
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            log.warning("failed to JPEG-encode frame_id=%s for job_id=%s", frame_id, job_id)
            return

        data = io.BytesIO(encoded.tobytes())
        key = frame_object_key(job_id, frame_id)
        try:
            self._minio.client.put_object(
                self._minio.bucket, key, data, length=len(encoded), content_type="image/jpeg"
            )
        except Exception:
            log.exception("failed to upload frame_id=%s for job_id=%s", frame_id, job_id)
