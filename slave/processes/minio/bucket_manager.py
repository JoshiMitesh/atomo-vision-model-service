"""Object-key layout for MinIO uploads.

Keeps every job's frames/videos isolated under its own job_id prefix
(CLAUDE.md §17):

    bucket/
    └── job_001/
        ├── frames/
        └── videos/
"""

from __future__ import annotations


def frame_object_key(job_id: str, frame_id: int) -> str:
    return f"{job_id}/frames/frame_{frame_id:08d}.jpg"  # just a simple function that takes a job_id and frame_id as input and returns a string representing the object key for a frame in the MinIO bucket. The object key is constructed by concatenating the job_id, "frames" directory, and the frame_id formatted as an 8-digit zero-padded number with a ".jpg" extension.


def video_object_key(job_id: str, segment_name: str) -> str:
    return f"{job_id}/videos/{segment_name}" # just a simple function that takes a job_id and segment_name as input and returns a string representing the object key for a video segment in the MinIO bucket. The object key is constructed by concatenating the job_id, "videos" directory, and the segment_name.
