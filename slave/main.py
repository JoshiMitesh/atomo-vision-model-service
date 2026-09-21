"""Slave Node entry point — composition root.

Run from the repository root (the parent of this `slave/` package):

    python -m slave.main
"""

from __future__ import annotations

import os

# Must happen before numpy/cv2 are imported anywhere in the process —
# both libraries read these once at load time. Without this, each
# spawns one thread per CPU core for its own internal parallelism
# (BLAS ops, OpenCV's parallel_for_), which on a multi-core board adds
# up to far more threads than cores once a job is running (NPU driver
# threads + RTSP decode + video encode + these all fighting for the
# same cores). Confirmed against this exact board's own working
# reference scripts (~/detection.py, ~/face_attrs_npu_rtsp.py) — our
# per-frame arrays (DFL/softmax/NMS) are small enough that
# single-threaded BLAS is not a bottleneck; multi-threading them was
# pure overhead.
for _env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")

import asyncio
import signal
import threading

import cv2

from slave.config import load_slave_config
from slave.logging.log_manager import LogManager
from slave.logging.logger import get_logger
from slave.processes.device.device_agent import DeviceAgent
from slave.processes.job.job_manager import JobManager
from slave.processes.job.job_runner import JobRunner
from slave.processes.minio.frame_uploader import FrameUploader
from slave.processes.minio.minio_client import MinioClientWrapper
from slave.processes.minio.video_uploader import VideoUploader
from slave.processes.model.model_manager import ModelManager
from slave.processes.model.model_registry import build_default_registry
from slave.processes.publisher.detection_publisher import DetectionPublisher
from slave.processes.publisher.publisher import Publisher
from slave.processes.publisher.upload_publisher import UploadPublisher
from slave.processes.subscriber.subscriber import NatsSubscriber
from slave.processes.worker.worker_factory import WorkerServices
from slave.processes.worker.worker_manager import WorkerManager
from slave.storage.yaml import job_store

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _bootstrap_jobs_yaml(path: str) -> None:
    if not os.path.exists(path):
        job_store.save_jobs(path, [])


def main() -> None:
    # Same reasoning as the env vars above — caps OpenCV's own internal
    # thread pool (parallel_for_, used by resize/rectangle/putText/
    # VideoWriter/VideoCapture decode), which is separate from BLAS and
    # not covered by the OMP/OPENBLAS/MKL env vars.

    # why thread count is 1 and what if use multiple threads?
    cv2.setNumThreads(1)

    config = load_slave_config()

    log_dir = config.resolve_path(config.logging.dir)
    LogManager.init(log_dir, config.logging.max_bytes, config.logging.backup_count, config.logging.level)
    log = get_logger(__name__)
    log.info("Slave node starting up")

    jobs_yaml_path = config.resolve_path(config.jobs.path)
    _bootstrap_jobs_yaml(jobs_yaml_path)

    # Shared, process-wide services.
    publisher = Publisher(config.nats) # creates a new publisher instance with the NATS configuration
    publisher.start() # start the publisher in a separate thread, which runs an asyncio event loop for NATS communication
    detection_publisher = DetectionPublisher(publisher, config.nats.detection_subject) # detection_publisher is an instance of the DetectionPublisher class, which is responsible for publishing detection results to the Master over NATS
    upload_publisher = UploadPublisher(publisher, config.nats.upload_subject) # upload_publisher is an instance of the UploadPublisher class, which is responsible for publishing upload events to the Master over NATS

    minio_client = MinioClientWrapper(config.minio) # creates a new MinioClientWrapper instance with the MinIO configuration, which is responsible for interacting with the MinIO object storage service
    frame_uploader = FrameUploader(minio_client) # creates a new FrameUploader instance with the MinioClientWrapper, responsible for uploading detected frames to MinIO. No longer takes upload_publisher — the frame's object_key/url are embedded directly into the DetectionResult by the worker instead of being published as a separate upload-started message.
    video_uploader = VideoUploader(minio_client, upload_publisher) # creates a new VideoUploader instance with the MinioClientWrapper and UploadPublisher instances, which is responsible for uploading completed 30-second video segments to MinIO and publishing upload events to the Master over NATS.

    services = WorkerServices(
        models_root=os.path.join(REPO_ROOT, "processes", "worker", "models"), # models_root is the path to the directory where the worker models are stored, which is used by the WorkerServices to load and manage the models for inference.
        config_root=config.resolve_path("config"), # config_root is the path to the directory where the configuration files are stored, which is used by the WorkerServices to load and manage the configuration for the workers.
        detection_publisher=detection_publisher, # detection_publisher is the instance of the DetectionPublisher class, which is used by the WorkerServices to publish detection results to the Master over NATS.
        frame_uploader=frame_uploader, # frame_uploader is the instance of the FrameUploader class, which is used by the WorkerServices to upload detected frames to MinIO and publish upload events to the Master over NATS.
        video_uploader=video_uploader, # video_uploader is the instance of the VideoUploader class, which is used by the WorkerServices to upload completed 30-second video segments to MinIO and publish upload events to the Master over NATS.
        video_temp_dir=config.resolve_path(config.video.temp_dir), # video_temp_dir is the path to the temporary directory where video segments are stored before being uploaded to MinIO, which is used by the WorkerServices to manage the temporary storage of video segments.
        video_segment_seconds=config.video.segment_seconds, # video_segment_seconds is the duration of each video segment in seconds, which is used by the WorkerServices to determine how long each video segment should be before it is uploaded to MinIO.
        video_fps=config.video.fps, # video_fps is the frames per second (FPS) of the video segments, which is used by the WorkerServices to determine the frame rate of the video segments before they are uploaded to MinIO.
        # source.branch_id / source.device_id on every cam.detection
        # message. branch_id is DeviceConfig's existing (previously
        # unused) site_id default; device_id is the real device UUID
        # already used for registration/telemetry.
        branch_id=config.device.site_id,
        device_id=config.device.device_id,
    )

    registry = build_default_registry() 
    worker_manager = WorkerManager()
    model_manager = ModelManager(registry, config.resolve_path("config"), services)
    job_manager = JobManager(worker_manager, model_manager)

    job_runner = JobRunner(jobs_yaml_path, config.jobs.poll_interval_seconds, job_manager)
    job_runner.start()

    # NATS subscriber runs its own asyncio loop in a background thread.
    # The Event must be constructed on subscriber_loop (inside the
    # subscriber thread, after set_event_loop), not on the main thread's
    # loop context — otherwise asyncio.Event.wait()/set() raise
    # "attached to a different loop" across threads.
    subscriber_loop = asyncio.new_event_loop()
    subscriber_stop_event: asyncio.Event | None = None

    def _subscriber_stop_setter() -> None:
        def _set() -> None:
            if subscriber_stop_event is not None:
                subscriber_stop_event.set()

        subscriber_loop.call_soon_threadsafe(_set)

    def _run_subscriber() -> None:
        nonlocal subscriber_stop_event
        asyncio.set_event_loop(subscriber_loop)
        subscriber_stop_event = asyncio.Event()
        subscriber_loop.run_until_complete(_subscriber_main())

    async def _subscriber_main() -> None:
        subscriber = NatsSubscriber(config.nats, jobs_yaml_path)
        await subscriber.start()
        try:
            await subscriber_stop_event.wait()
        finally:
            await subscriber.stop()

    subscriber_thread = threading.Thread(target=_run_subscriber, name="nats-subscriber", daemon=True)
    subscriber_thread.start()

    # Device health/telemetry/command channel — independent of the job
    # pipeline above (see processes/device/device_agent.py).
    device_agent = DeviceAgent(config.nats, config.device, jobs_yaml_path)
    device_agent.start()

    shutdown_event = threading.Event()

    def _handle_signal(signum, frame) -> None:
        log.info("received signal %s, shutting down", signum)
        shutdown_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log.info("Slave node running (person detection only)")
    shutdown_event.wait()

    log.info("Slave node shutting down")
    _subscriber_stop_setter()
    subscriber_thread.join(timeout=10)
    device_agent.stop()
    job_runner.stop()
    worker_manager.stop_all()
    publisher.stop()
    log.info("Slave node shutdown complete")


if __name__ == "__main__":
    main()
