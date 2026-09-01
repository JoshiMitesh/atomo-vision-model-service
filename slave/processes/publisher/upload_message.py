"""Build the JSON-safe NATS payload for a MinIO upload-started event.

Fired the moment an upload to MinIO begins (not on completion) so
Master learns where a frame/video will live as soon as the Slave
starts sending it — see processes/minio/frame_uploader.py and
processes/minio/video_uploader.py, the only two callers.
"""

from __future__ import annotations

from typing import Optional


def build_upload_message(
    job_id: str,
    kind: str,
    object_key: str,
    url: str,
    frame_id: Optional[int] = None,
    segment_name: Optional[str] = None,
) -> dict:
    message: dict = {
        "job_id": job_id,
        "kind": kind,  # "frame" | "video"
        "object_key": object_key,
        "url": url,
    }
    if frame_id is not None:
        message["frame_id"] = frame_id
    if segment_name is not None:
        message["segment_name"] = segment_name
    return message
