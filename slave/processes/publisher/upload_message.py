"""Build the JSON-safe NATS payload for a MinIO upload-started event.

Fired the moment an upload to MinIO begins (not on completion) so
Master learns where a frame/video will live as soon as the Slave
starts sending it — see processes/minio/frame_uploader.py and
processes/minio/video_uploader.py, the only two callers.
"""

from __future__ import annotations

from typing import Optional

# built upload message for NATS, to be sent to Master when an upload starts
def build_upload_message(
    job_id: str,
    kind: str,
    object_key: str,
    url: str,
    frame_id: Optional[int] = None,
    segment_name: Optional[str] = None,
) -> dict:   # builds a dictionary representing the upload-started event message, which includes the job ID, kind of upload (frame or video), object key, URL, and optionally frame ID and segment name. This dictionary is JSON-safe and can be published over NATS.
    message: dict = {
        "job_id": job_id,
        "kind": kind,  # "frame" | "video"
        "object_key": object_key,
        "url": url,
    } # message is a dictionary that contains the basic information about the upload event, including the job ID, kind of upload (frame or video), object key, and URL. Optional fields for frame ID and segment name are added if provided.
    if frame_id is not None:
        message["frame_id"] = frame_id # frame_id is an optional integer that represents the ID of the frame being uploaded. If provided, it is added to the message dictionary.
    if segment_name is not None:
        message["segment_name"] = segment_name # segment_name is an optional string that represents the name of the video segment being uploaded. If provided, it is added to the message dictionary.
    return message
