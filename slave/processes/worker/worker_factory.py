"""Instantiates the concrete Person/Face/Safety worker for a resolved job.

Pure construction — no business logic lives here (CLAUDE.md §22: "do
not put all model logic in worker_factory.py"). model_manager.py looks
up *which* worker class to build via model_registry.py; this module
only wires a job's config and the process-wide shared services into
that class.

Person AND Safety jobs each get an extra face_controller (the
auto-triggered Face detector — config/models/face.yaml's
`trigger.enabled` toggles it on/off for BOTH model types, one shared
flag). Person's worker gates the trigger on its own person detections;
Safety's worker gates it on just its own "person" class among its 9
classes (see safety_*_worker.py). Standalone Face jobs (model_type:
"face") don't take a face_controller at all — they run their own Face
detector directly, unconditionally, on their own stream/job (see
processes/worker/models/face/{pt,nb,so}/face_*_worker.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Type

from slave.models.job import JobConfig, ModelAssignment
from slave.processes.minio.frame_uploader import FrameUploader
from slave.processes.minio.video_uploader import VideoUploader
from slave.processes.model.model_resolver import ResolvedModel
from slave.processes.publisher.detection_publisher import DetectionPublisher
from slave.processes.publisher.profiling_publisher import ProfilingPublisher
from slave.processes.video.video_manager import VideoManager
from slave.processes.worker.face.face_factory import build_face_controller


def worker_key_for(job_id: str, model_type: str) -> str:
    """The internal isolation key one assignment's worker uses for
    everything that must never collide with a sibling assignment's
    worker under the same job_id — its RTSP FrameReader, its video
    temp-dir/segment naming, and its MinIO frame/video object keys.
    Master-facing messages (DetectionResult.job_id) keep using the
    real job_id untouched; only this internal plumbing uses the
    composite key.
    """
    return f"{job_id}__{model_type}"


@dataclass(frozen=True)
class WorkerIdentity:
    """Message-envelope identity shared by every worker (person or
    face) — the parts of cam.detection's `source` object that come
    from this device/config rather than from a per-frame detection.
    branch_id/branch_name are NOT here — they're per-job, Master-
    supplied (see JobConfig), not device-level.
    """

    device_id: str
    device_name: str


@dataclass(frozen=True)
class WorkerServices:
    """Process-wide collaborators shared (read-only) across all jobs."""

    models_root: str
    config_root: str
    detection_publisher: DetectionPublisher
    frame_uploader: FrameUploader
    video_uploader: VideoUploader
    video_temp_dir: str
    video_segment_seconds: int
    video_fps: int
    device_id: str
    device_name: str
    profiling_publisher: ProfilingPublisher  # new field for profiling telemetry


def create_worker(
    job: JobConfig,
    assignment: ModelAssignment,
    resolved: ResolvedModel,
    worker_cls: Type,
    model_config: dict,
    services: WorkerServices,
):
    models_dir = os.path.join(services.models_root, resolved.model_type)
    worker_key = worker_key_for(job.job_id, assignment.model_type)

    video_manager = VideoManager(
        job_id=worker_key,
        video_uploader=services.video_uploader,
        temp_dir=services.video_temp_dir,
        segment_seconds=services.video_segment_seconds,
        fps=services.video_fps,
    )

    identity = WorkerIdentity(
        device_id=services.device_id,
        device_name=services.device_name,
    )

    common_kwargs = dict(
        job=job,
        assignment=assignment,
        worker_key=worker_key,
        model_config=model_config,
        models_dir=models_dir,
        detection_publisher=services.detection_publisher,
        frame_uploader=services.frame_uploader,
        video_manager=video_manager,
        identity=identity,
        profiling_publisher=services.profiling_publisher,  # profiling telemetry publisher
    )

    if resolved.model_type in ("person", "safety"):
        # Face detection is auto-triggered by the Person/Safety worker
        # itself, on the same stream/frame — it is never a separate
        # jobs.yaml entry for this path, so it's wired in here rather
        # than through model_registry.py. Both share the same
        # config/models/face.yaml trigger.enabled flag; each worker's
        # own _process_frame() decides what counts as "triggered"
        # (Person: any person detection; Safety: its own "person"
        # class only).
        face_controller = build_face_controller(
            job_id=worker_key,
            model_extension=resolved.model_extension,
            models_root=services.models_root,
            config_root=services.config_root,
        )
        return worker_cls(face_controller=face_controller, **common_kwargs)

    return worker_cls(**common_kwargs)
