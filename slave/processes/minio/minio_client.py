"""Thin wrapper around the MinIO SDK client + bucket bootstrap."""

from __future__ import annotations

from minio import Minio

from slave.config import MinioConfig
from slave.logging.logger import get_logger

log = get_logger(__name__)


class MinioClientWrapper:   # minio
    def __init__(self, config: MinioConfig) -> None:
        self._config = config # takes MinioConfig object as input, which contains endpoint, access_key, secret_key, secure, and bucket information
        self._client: Minio | None = None # only constructed when minio.enabled=true — see config.enabled branch below
        if config.enabled:
            self._client = Minio(
                config.endpoint,
                access_key=config.access_key,
                secret_key=config.secret_key,
                secure=config.secure,
            )  # takes the endpoint, access_key, secret_key, and secure flag from the config and creates a Minio client object
            self._ensure_bucket() # calls the _ensure_bucket method to check if the specified bucket exists, and creates it if it doesn't
        else:
            log.info("MinIO disabled (minio.enabled=false) — uploads will be skipped") # config.enabled=false — skip client construction and bucket bootstrap entirely

    def _ensure_bucket(self) -> None:  # ensures the bucket exists
        if not self._client.bucket_exists(self._config.bucket): # checks if the bucket specified in the config exists
            self._client.make_bucket(self._config.bucket) # creates the bucket if it doesn't exist
            log.info("created MinIO bucket '%s'", self._config.bucket) # logs an info message indicating that the bucket was created

    @property
    def enabled(self) -> bool:
        return self._config.enabled # exposes config.enabled so callers (frame/video uploaders) can skip uploads without touching self._client directly

    @property
    def bucket(self) -> str:
        return self._config.bucket # returns the name of the bucket specified in the config

    @property
    def client(self) -> Minio:
        return self._client  # returns the Minio client object

    def object_url(self, key: str) -> str: # returns the URL for a given object key in the MinIO bucket
        scheme = "https" if self._config.secure else "http" # determines the URL scheme based on the secure flag in the config
        return f"{scheme}://{self._config.endpoint}/{self._config.bucket}/{key}" # constructs and returns the full URL for the object in the MinIO bucket using the scheme, endpoint, bucket name, and object key
