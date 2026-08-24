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
    cv2.setNumThreads(1)

    config = load_slave_config()

    log_dir = config.resolve_path(config.logging.dir)
    LogManager.init(log_dir, config.logging.max_bytes, config.logging.backup_count, config.logging.level)
    log = get_logger(__name__)
    log.info("Slave node starting up")

    jobs_yaml_path = config.resolve_path(config.jobs.path)
    _bootstrap_jobs_yaml(jobs_yaml_path)

    # Shared, process-wide services.
    publisher = Publisher(config.nats)
    publisher.start()
    detection_publisher = DetectionPublisher(publisher, config.nats.detection_subject)
    upload_publisher = UploadPublisher(publisher, config.nats.upload_subject)

    minio_client = MinioClientWrapper(config.minio)
    frame_uploader = FrameUploader(minio_client, upload_publisher)
    video_uploader = VideoUploader(minio_client, upload_publisher)

    services = WorkerServices(
        models_root=os.path.join(REPO_ROOT, "processes", "worker", "models"),
        config_root=config.resolve_path("config"),
        detection_publisher=detection_publisher,
        frame_uploader=frame_uploader,
        video_uploader=video_uploader,
        video_temp_dir=config.resolve_path(config.video.temp_dir),
        video_segment_seconds=config.video.segment_seconds,
        video_fps=config.video.fps,
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
    device_agent = DeviceAgent(config.nats, config.device)
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
