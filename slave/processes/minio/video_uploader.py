"""Uploads completed 30-second video segments to MinIO."""

from __future__ import annotations

from slave.logging.logger import get_logger
from slave.processes.minio.bucket_manager import video_object_key
from slave.processes.minio.minio_client import MinioClientWrapper
from slave.utils.file_utils import safe_remove

log = get_logger(__name__)


class VideoUploader:
    def __init__(self, minio_client: MinioClientWrapper) -> None:
        self._minio = minio_client

    def upload_video_segment(self, job_id: str, local_path: str, segment_name: str) -> None:
        key = video_object_key(job_id, segment_name)
        try:
            self._minio.client.fput_object(self._minio.bucket, key, local_path, content_type="video/mp4")
            log.info("uploaded video segment '%s' for job_id=%s", segment_name, job_id)
        except Exception:
            log.exception("failed to upload video segment '%s' for job_id=%s", segment_name, job_id)
        finally:
            safe_remove(local_path)
