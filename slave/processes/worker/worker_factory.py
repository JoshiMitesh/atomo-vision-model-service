"""Instantiates the concrete Person worker for a resolved job.

Pure construction — no business logic lives here (CLAUDE.md §22: "do
not put all model logic in worker_factory.py"). model_manager.py looks
up *which* worker class to build via model_registry.py; this module
only wires a job's config and the process-wide shared services into
that class.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Type

from slave.models.job import JobConfig
from slave.processes.minio.frame_uploader import FrameUploader
from slave.processes.minio.video_uploader import VideoUploader
from slave.processes.model.model_resolver import ResolvedModel
from slave.processes.publisher.detection_publisher import DetectionPublisher
from slave.processes.video.video_manager import VideoManager
from slave.processes.worker.face.face_factory import build_face_controller


@dataclass(frozen=True)
class WorkerServices:
    """Process-wide collaborators shared (read-only) across all jobs."""

    models_root: str  # .../processes/worker/models
    config_root: str  # .../config — needed to resolve config/models/face.yaml
    detection_publisher: DetectionPublisher
    frame_uploader: FrameUploader
    video_uploader: VideoUploader
    video_temp_dir: str
    video_segment_seconds: int
    video_fps: int


def create_worker(
    job: JobConfig,
    resolved: ResolvedModel,
    worker_cls: Type,
    model_config: dict,
    services: WorkerServices,
):
    models_dir = os.path.join(services.models_root, resolved.model_type)

    video_manager = VideoManager(
        job_id=job.job_id,
        video_uploader=services.video_uploader,
        temp_dir=services.video_temp_dir,
        segment_seconds=services.video_segment_seconds,
        fps=services.video_fps,
    )

    # Face detection is auto-triggered by the Person worker itself, on
    # the same stream/frame — it is never a separate jobs.yaml entry,
    # so it's wired in here rather than through model_registry.py.
    face_controller = build_face_controller(
        job_id=job.job_id,
        model_extension=resolved.model_extension,
        models_root=services.models_root,
        config_root=services.config_root,
    )

    return worker_cls(
        job=job,
        model_config=model_config,
        models_dir=models_dir,
        detection_publisher=services.detection_publisher,
        frame_uploader=services.frame_uploader,
        video_manager=video_manager,
        face_controller=face_controller,
    )
