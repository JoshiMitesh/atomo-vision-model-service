"""Resolves a job's camera_id into a connectable RTSP stream URL.

STATUS: PLACEHOLDER. The job-start message carries only `camera_id`
(no master_ip/stream_port/stream_name anymore) — the real mapping from
camera_id to an actual stream lives on Master's side (it's whatever
port+name that camera was registered with in the frontend), and how
Slave is meant to obtain it from Master was not settled as of this
writing: not a URL on the job message, not a fixed host/port formula,
possibly a NATS request/reply or a separate camera-registry push.

Until that's confirmed, this builds a best-effort URL using camera_id
as the RTSP path against the existing NATS broker host (slave.yaml's
nats.host — no new config was introduced) and a hardcoded default
port. Replace build_stream_url()'s body with the real resolution once
Master's side of this is confirmed; every caller (Person/Face workers)
only depends on this function's signature, not its internals, so the
swap is contained to this one file.
"""

from __future__ import annotations

# TODO(camera resolution): placeholder only, see module docstring.
DEFAULT_STREAM_PORT = 8554


def build_stream_url(stream_host: str, camera_id: str, port: int = DEFAULT_STREAM_PORT) -> str:
    return f"rtsp://{stream_host}:{port}/{camera_id}"
