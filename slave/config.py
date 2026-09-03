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
class NatsConfig: # NatsConfig is a dataclass that holds the configuration for connecting to a NATS server, including the host, port, subjects for job control, detection results, and upload events, as well as optional user credentials and reconnect wait time.
    host: str
    port: int
    job_subject: str
    stop_subject: str
    detection_subject: str
    upload_subject: str
    reconnect_wait_seconds: float = 2.0
    user: Optional[str] = None
    password: Optional[str] = None
    use_websocket: bool = False

    @property
    def url(self) -> str:
        scheme = "ws" if self.use_websocket else "nats" # picks the ws:// scheme when nats.use_websocket=true (config/slave.yaml), otherwise the plain nats:// scheme
        return f"{scheme}://{self.host}:{self.port}" # url is a property that returns the NATS server URL in the format "scheme://host:port", which can be used to connect to the NATS server.


@dataclass(frozen=True)
class JobsConfig: # JobsConfig is a dataclass that holds the configuration for job management, including the path to the jobs YAML file and the polling interval in seconds for checking for new jobs.
    path: str
    poll_interval_seconds: int = 60


@dataclass(frozen=True)
class MinioConfig: # MinioConfig is a dataclass that holds the configuration for connecting to a MinIO object storage service, including the endpoint, access key, secret key, bucket name, and secure flag.
    endpoint: str
    access_key: str
    secret_key: str
    bucket: str
    secure: bool = False
    enabled: bool = True # when false, MinioClientWrapper skips client construction entirely and frame/video uploaders no-op — see minio_client.py


@dataclass(frozen=True)
class VideoConfig: # VideoConfig is a dataclass that holds the configuration for video processing, including the duration of each video segment in seconds, the temporary directory for storing video segments, and the frames per second (FPS) of the video segments.
    segment_seconds: int = 30
    temp_dir: str = "storage/tmp/video"
    fps: int = 15


@dataclass(frozen=True)
class LoggingConfig: # LoggingConfig is a dataclass that holds the configuration for logging, including the directory for log files, the maximum size of each log file in bytes, the number of backup log files to keep, and the logging level.
    dir: str = "logging_output"
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5
    level: str = "INFO"


@dataclass(frozen=True) # device_config is a dataclass that holds the configuration for device health, telemetry, and command channels, including the site ID, slave ID, data stream name, data interval in seconds, health bucket name, health time-to-live (TTL) in seconds, and heartbeat interval in seconds.
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


@dataclass(frozen=True) # slave_config is a dataclass that holds the overall configuration for the slave node, including the NATS configuration, jobs configuration, MinIO configuration, video configuration, logging configuration, device configuration, and the base directory for resolving relative paths.
class SlaveConfig:
    nats: NatsConfig
    jobs: JobsConfig
    minio: MinioConfig
    video: VideoConfig
    logging: LoggingConfig
    device: DeviceConfig
    _base_dir: str = field(default=REPO_ROOT) # _base_dir is a private field that holds the base directory for resolving relative paths in the configuration. It defaults to the repository root directory, but can be overridden if needed.

    def resolve_path(self, relative_path: str) -> str: # resolve path is a method that takes a relative path as input and resolves it against the base directory of the configuration. If the input path is already absolute, it returns it unchanged; otherwise, it joins the base directory with the relative path to produce an absolute path.
        """Resolve a config-relative path against the repo root."""
        if os.path.isabs(relative_path): # checks if the input relative_path is already an absolute path using os.path.isabs(). If it is, the method returns it unchanged.
            return relative_path
        return os.path.join(self._base_dir, relative_path) # If the input relative_path is not absolute, the method joins the base directory (_base_dir) with the relative_path using os.path.join() to produce an absolute path, which it then returns.


def load_slave_config(path: str | None = None) -> SlaveConfig: # load_slave_config is a function that loads and validates the configuration from the specified slave.yaml file (or the default path if none is provided) and returns a SlaveConfig object containing the parsed configuration data. It raises a ValueError if any required sections are missing from the YAML file.
    """Load and validate config/slave.yaml into a SlaveConfig."""
    config_path = path or os.path.join(REPO_ROOT, "config", "slave.yaml") # config_path is the path to the slave.yaml configuration file. If a path is provided as an argument, it uses that; otherwise, it defaults to "config/slave.yaml" in the repository root directory.
    with open(config_path, "r", encoding="utf-8") as fh: # opens the slave.yaml file for reading with UTF-8 encoding and assigns the file handle to fh.
        raw = yaml.safe_load(fh) or {} # yaml.safe_load(fh) reads the contents of the file and parses it as YAML, returning a Python dictionary. If the file is empty or cannot be parsed, it defaults to an empty dictionary.

    try:
        nats_raw = raw["nats"] # extracts the "nats" section from the parsed YAML dictionary and assigns it to nats_raw. This section contains the configuration for connecting to the NATS server.
        jobs_raw = raw["jobs"] # extracts the "jobs" section from the parsed YAML dictionary and assigns it to jobs_raw. This section contains the configuration for job management.
        minio_raw = raw["minio"] # extracts the "minio" section from the parsed YAML dictionary and assigns it to minio_raw. This section contains the configuration for connecting to the MinIO object storage.
    except KeyError as exc:
        raise ValueError(f"slave.yaml is missing required section: {exc}") from exc  # raises a ValueError with a message indicating which required section is missing from the slave.yaml file if any of the expected sections ("nats", "jobs", or "minio") are not found in the parsed YAML dictionary. The original KeyError is chained to provide context for the error.

    return SlaveConfig( # returns a new instance of the SlaveConfig dataclass, initialized with the parsed configuration data from the YAML file. Each section of the configuration is passed to the corresponding dataclass constructor (NatsConfig, JobsConfig, MinioConfig, VideoConfig, LoggingConfig, DeviceConfig), and the base directory is set to REPO_ROOT.
        nats=NatsConfig( # NatsConfig is initialized with the values extracted from the "nats" section of the YAML file, including host, port, subjects for job control, detection results, and upload events, as well as optional user credentials and reconnect wait time.
            host=nats_raw["host"],
            port=int(nats_raw["port"]),
            job_subject=nats_raw["job_subject"],
            stop_subject=nats_raw["stop_subject"],
            detection_subject=nats_raw["detection_subject"],
            upload_subject=nats_raw["upload_subject"],
            reconnect_wait_seconds=float(nats_raw.get("reconnect_wait_seconds", 2.0)),
            use_websocket=bool(nats_raw.get("use_websocket", False)), # reads nats.use_websocket from slave.yaml (default false) — flips NatsConfig.url between ws:// and nats://
        ),
        jobs=JobsConfig( # JobsConfig is initialized with the values extracted from the "jobs" section of the YAML file, including the path to the jobs directory and the polling interval.
            path=jobs_raw["path"],
            poll_interval_seconds=int(jobs_raw.get("poll_interval_seconds", 60)),
        ),
        minio=MinioConfig( # MinioConfig is initialized with the values extracted from the "minio" section of the YAML file, including the endpoint, access key, secret key, and bucket information.
            endpoint=minio_raw["endpoint"],
            access_key=minio_raw["access_key"],
            secret_key=minio_raw["secret_key"],
            bucket=minio_raw["bucket"],
            secure=bool(minio_raw.get("secure", False)),
            enabled=bool(minio_raw.get("enabled", True)), # reads minio.enabled from slave.yaml (default true) — set false to skip MinIO client construction and all uploads
        ),
        video=VideoConfig(**(raw.get("video") or {})), # VideoConfig is initialized with the values extracted from the "video" section of the YAML file, or defaults if the section is missing. It includes the duration of each video segment, the temporary directory for storing video segments, and the frames per second (FPS) of the video segments.
        logging=LoggingConfig(**(raw.get("logging") or {})), # LoggingConfig is initialized with the values extracted from the "logging" section of the YAML file, or defaults if the section is missing. It includes the directory for log files, the maximum size of each log file, the number of backup log files to keep, and the logging level.
        device=DeviceConfig(**(raw.get("device") or {})), # DeviceConfig is initialized with the values extracted from the "device" section of the YAML file, or defaults if the section is missing. It includes the site ID, slave ID, data stream name, data interval, health bucket, health TTL, and heartbeat interval for device health and telemetry.
        _base_dir=REPO_ROOT, # _base_dir is set to the repository root directory, which is used for resolving relative paths in the configuration. This allows the configuration to reference files and directories relative to the root of the repository.
    )
