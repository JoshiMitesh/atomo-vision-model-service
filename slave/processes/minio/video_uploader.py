"""Uploads completed 30-second video segments to MinIO."""

from __future__ import annotations

from slave.logging.logger import get_logger
from slave.processes.minio.bucket_manager import video_object_key
from slave.processes.minio.minio_client import MinioClientWrapper
from slave.processes.publisher.upload_publisher import UploadPublisher
from slave.utils.file_utils import safe_remove

log = get_logger(__name__) # get_logger is a function that returns a logger instance for the current module, which can be used to log messages at different levels (info, warning, error, etc.)


class VideoUploader: # VideoUploader is a class that handles the uploading of completed 30-second video segments to MinIO and publishes upload events to the Master over NATS.
    def __init__(self, minio_client: MinioClientWrapper, upload_publisher: UploadPublisher) -> None: # initializes the VideoUploader with a MinioClientWrapper and an UploadPublisher
        self._minio = minio_client # stores the MinioClientWrapper instance for uploading video segments to MinIO
        self._upload_publisher = upload_publisher # stores the UploadPublisher instance for publishing upload events to the Master over NATS

    def upload_video_segment(self, job_id: str, local_path: str, segment_name: str) -> None: # uploads a completed 30-second video segment to MinIO and publishes an upload-started event to the Master over NATS. It takes the job_id, local_path of the video segment, and the segment_name as input.
        if not self._minio.enabled:
            # Still clean up the finished local segment file — CLAUDE.md
            # §18: a video failure/skip must not leave temp files behind
            # or corrupt jobs.yaml.
            safe_remove(local_path)
            return

        key = video_object_key(job_id, segment_name) # key is the object key for the video segment in the MinIO bucket, constructed using the job_id and segment_name. The video_object_key function generates the appropriate object key based on the job_id and segment_name.

        # Announce the upload to Master as it starts, not on completion —
        # gives Master the MinIO link immediately (CLAUDE.md request).
        self._upload_publisher.publish_upload_started( # publishes an upload-started event to the Master over NATS using the UploadPublisher instance. The event includes the job_id, kind (video), object_key, URL of the uploaded video segment in MinIO, and the segment_name.
            job_id=job_id,
            kind="video",
            object_key=key,
            url=self._minio.object_url(key),
            segment_name=segment_name,
        )

        try:
            self._minio.client.fput_object(self._minio.bucket, key, local_path, content_type="video/mp4") # uploads the video segment to MinIO using the MinioClientWrapper instance. The fput_object method takes the bucket name, object key, local file path, and content type (video/mp4) as parameters.
            log.info("uploaded video segment '%s' for job_id=%s", segment_name, job_id) # log an info message indicating that the video segment was successfully uploaded to MinIO
        except Exception:
            log.exception("failed to upload video segment '%s' for job_id=%s", segment_name, job_id) # log the exception if the video segment upload to MinIO fails
        finally:
            safe_remove(local_path) # safely removes the local video segment file after the upload attempt, regardless of whether the upload was successful or not. This helps to free up disk space and avoid cluttering the local filesystem with temporary video files.
