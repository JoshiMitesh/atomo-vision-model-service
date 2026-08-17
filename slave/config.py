"""Typed access to config/slave.yaml.

This module owns *process*-level configuration (NATS, MinIO, logging,
polling interval, video segmenting). Model-specific configuration
(config/models/person.yaml) is loaded separately by
``processes/model/model_resolver.py`` — it is not mixed in here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import yaml

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class NatsConfig:
    host: str
    port: int
    job_subject: str
    stop_subject: str
    detection_subject: str
    reconnect_wait_seconds: float = 2.0
    user: Optional[str] = None
    password: Optional[str] = None

    @property
    def url(self) -> str:
        return f"nats://{self.host}:{self.port}"


@dataclass(frozen=True)
class JobsConfig:
    path: str
    poll_interval_seconds: int = 60


@dataclass(frozen=True)
class MinioConfig:
    endpoint: str
    access_key: str
    secret_key: str
    bucket: str
    secure: bool = False


@dataclass(frozen=True)
class VideoConfig:
    segment_seconds: int = 30
    temp_dir: str = "storage/tmp/video"
    fps: int = 15


@dataclass(frozen=True)
class LoggingConfig:
    dir: str = "logging_output"
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5
    level: str = "INFO"


@dataclass(frozen=True)
class DeviceConfig:
    """Identity + timing for the device health/telemetry/command
    channel (processes/device/) — separate from job control.
    """

    site_id: str = "site1"
    slave_id: str = "slave01"
    data_stream_name: str = "DEVICE_DATA"
    data_interval_seconds: float = 30.0
    health_bucket: str = "device_health"
    health_ttl_seconds: int = 30
    heartbeat_interval_seconds: float = 5.0


@dataclass(frozen=True)
class SlaveConfig:
    nats: NatsConfig
    jobs: JobsConfig
    minio: MinioConfig
    video: VideoConfig
    logging: LoggingConfig
    device: DeviceConfig
    _base_dir: str = field(default=REPO_ROOT)

    def resolve_path(self, relative_path: str) -> str:
        """Resolve a config-relative path against the repo root."""
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(self._base_dir, relative_path)


def load_slave_config(path: str | None = None) -> SlaveConfig:
    """Load and validate config/slave.yaml into a SlaveConfig."""
    config_path = path or os.path.join(REPO_ROOT, "config", "slave.yaml")
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    try:
        nats_raw = raw["nats"]
        jobs_raw = raw["jobs"]
        minio_raw = raw["minio"]
    except KeyError as exc:
        raise ValueError(f"slave.yaml is missing required section: {exc}") from exc

    return SlaveConfig(
        nats=NatsConfig(
            host=nats_raw["host"],
            port=int(nats_raw["port"]),
            job_subject=nats_raw["job_subject"],
            stop_subject=nats_raw["stop_subject"],
            detection_subject=nats_raw["detection_subject"],
            reconnect_wait_seconds=float(nats_raw.get("reconnect_wait_seconds", 2.0)),
            user=nats_raw.get("user"),
            password=nats_raw.get("password"),
        ),
        jobs=JobsConfig(
            path=jobs_raw["path"],
            poll_interval_seconds=int(jobs_raw.get("poll_interval_seconds", 60)),
        ),
        minio=MinioConfig(
            endpoint=minio_raw["endpoint"],
            access_key=minio_raw["access_key"],
            secret_key=minio_raw["secret_key"],
            bucket=minio_raw["bucket"],
            secure=bool(minio_raw.get("secure", False)),
        ),
        video=VideoConfig(**(raw.get("video") or {})),
        logging=LoggingConfig(**(raw.get("logging") or {})),
        device=DeviceConfig(**(raw.get("device") or {})),
        _base_dir=REPO_ROOT,
    )
