"""Uploads detected frames to MinIO.

This lives outside the detectors on purpose (CLAUDE.md §17, §22) —
detectors only produce Detection objects, workers decide whether/what
to upload.

Does NOT publish its own "upload started" NATS message anymore — the
object_key/url this returns get embedded directly into the
DetectionResult by the worker instead, merging what used to be two
separate messages (detection + frame-upload) into one per frame. Video
segment uploads are unrelated to a single frame/detection, so
video_uploader.py keeps its own separate UploadPublisher message.
"""

from __future__ import annotations

import io
from typing import Optional, Tuple

import cv2
import numpy as np

from slave.logging.logger import get_logger
from slave.processes.minio.bucket_manager import face_object_key, frame_object_key
from slave.processes.minio.minio_client import MinioClientWrapper

log = get_logger(__name__)


class FrameUploader:
    def __init__(self, minio_client: MinioClientWrapper) -> None:
        self._minio = minio_client

    def upload_frame(self, job_id: str, frame_id: int, image: np.ndarray) -> Optional[Tuple[str, str]]:
        """Returns (object_key, url) on success, None if MinIO is
        disabled, encoding failed, or the upload itself failed.
        """
        if not self._minio.enabled:
            return None

        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            log.warning("failed to JPEG-encode frame_id=%s for job_id=%s", frame_id, job_id)
            return None

        data = io.BytesIO(encoded.tobytes())
        key = frame_object_key(job_id, frame_id)
        url = self._minio.object_url(key)

        try:
            self._minio.client.put_object(
                self._minio.bucket, key, data, length=len(encoded), content_type="image/jpeg"
            )
        except Exception:
            log.exception("failed to upload frame_id=%s for job_id=%s", frame_id, job_id)
            return None

        return key, url

    def upload_face_crop(
        self,
        job_id: str,
        frame_id: int,
        track_id: str,
        image: np.ndarray,
    ) -> Optional[Tuple[str, str]]:
        """Upload one face crop and return its MinIO object key and URL."""
        if not self._minio.enabled:
            return None

        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            log.warning(
                "failed to JPEG-encode face crop frame_id=%s track_id=%s job_id=%s",
                frame_id,
                track_id,
                job_id,
            )
            return None

        data = io.BytesIO(encoded.tobytes())
        key = face_object_key(job_id, track_id, frame_id)
        url = self._minio.object_url(key)

        try:
            self._minio.client.put_object(
                self._minio.bucket,
                key,
                data,
                length=len(encoded),
                content_type="image/jpeg",
            )
        except Exception:
            log.exception(
                "failed to upload face crop frame_id=%s track_id=%s job_id=%s",
                frame_id,
                track_id,
                job_id,
            )
            return None

        log.info("uploaded face crop object_key=%s", key)
        return key, url
