"""Thin wrapper around the MinIO SDK client + bucket bootstrap."""

from __future__ import annotations

from minio import Minio

from slave.config import MinioConfig
from slave.logging.logger import get_logger

log = get_logger(__name__)


class MinioClientWrapper:
    def __init__(self, config: MinioConfig) -> None:
        self._config = config
        self._client = Minio(
            config.endpoint,
            access_key=config.access_key,
            secret_key=config.secret_key,
            secure=config.secure,
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._config.bucket):
            self._client.make_bucket(self._config.bucket)
            log.info("created MinIO bucket '%s'", self._config.bucket)

    @property
    def bucket(self) -> str:
        return self._config.bucket

    @property
    def client(self) -> Minio:
        return self._client
