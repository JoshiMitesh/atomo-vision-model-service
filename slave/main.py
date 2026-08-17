"""Slave Node entry point — composition root.

Run from the repository root (the parent of this `slave/` package):

    python -m slave.main
"""

from __future__ import annotations

import asyncio
import os
import signal
import threading

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
from slave.processes.subscriber.subscriber import NatsSubscriber
from slave.processes.worker.worker_factory import WorkerServices
from slave.processes.worker.worker_manager import WorkerManager
from slave.storage.yaml import job_store

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _bootstrap_jobs_yaml(path: str) -> None:
    if not os.path.exists(path):
        job_store.save_jobs(path, [])


def main() -> None:
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

    minio_client = MinioClientWrapper(config.minio)
    frame_uploader = FrameUploader(minio_client)
    video_uploader = VideoUploader(minio_client)

    services = WorkerServices(
        models_root=os.path.join(REPO_ROOT, "processes", "worker", "models"),
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
    subscriber_stop_event = asyncio.Event()
    subscriber_loop = asyncio.new_event_loop()

    def _subscriber_stop_setter() -> None:
        subscriber_loop.call_soon_threadsafe(subscriber_stop_event.set)

    def _run_subscriber() -> None:
        asyncio.set_event_loop(subscriber_loop)
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
