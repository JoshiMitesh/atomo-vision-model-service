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
    return f"{job_id}/frames/frame_{frame_id:08d}.jpg"


def video_object_key(job_id: str, segment_name: str) -> str:
    return f"{job_id}/videos/{segment_name}"
