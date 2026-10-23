"""Standalone Number-Plate `.nb` worker -- a jobs.yaml entry with
model_type: "number_plate", independent of any Person/Safety job.

Uses the shared SchedulerBackedWorker mixin
(runtimes/common/round_robin_scheduler.py) instead of owning its own
NbRuntime handle + thread -- the A311D NPU can only run one model
instance at a time, and independent per-job handles deadlock once two
or more Number-Plate `.nb` jobs are active together (confirmed live:
two concurrent jobs each got exactly one successful NPU inference,
then both went silent for the rest of the run). Every job using the
same number_plate.nb file now shares one handle and takes turns.
start()/stop()/is_alive()/the detections callback are all inherited
from the mixin -- only _attach_scheduler() (in __init__),
_process_frame(), and _cleanup() are written here.

Per-job PlateEnricher (plate_enricher.py: tracking + OCR + plate-
number voting -- the SAME module the `.pt` path uses, not a separate
NPU-specific port) stays per-job, never shared -- a different
camera's plate history must never mix with another's (CLAUDE.md SS20).
Only `detect()` (the raw box detector) and the NPU handle itself are
shared, because the hardware forces it.

Used to have its own npu_plate_enricher.py/npu_plate_ocr.py/
npu_plate_tracker.py, independently written and independently tuned
from the `.pt` path's. That duplication was the direct cause of two
bugs live-testing found: npu_plate_ocr.py's perspective-correction
fallback silently truncated the last character of every read of a
plate (e.g. "GJ11S7924" published as "GJ11S792", every single time --
.pt's plate_ocr.py never warped the crop at all and never had this
bug), and npu_plate_enricher.py's confirmation threshold had quietly
drifted to require 3 matching reads against .pt's 2, a stricter bar
nobody had decided on, just never kept in sync. Merging onto the one
proven implementation removes the possibility of that kind of drift
recurring, and gives `.nb` `.pt`'s full 4-way OCR voting (2 pad ratios
x raw/enhanced) instead of the 2-way vote npu_plate_ocr.py had.
"""

from __future__ import annotations

import os

from slave.logging.logger import get_logger
from slave.models.detection import group_detections
from slave.models.job import JobConfig, ModelAssignment
from slave.processes.detection.detection_result import DetectionResult
from slave.processes.detection.frame_annotator import draw_detections
from slave.processes.minio.frame_uploader import FrameUploader
from slave.processes.publisher.detection_publisher import DetectionPublisher
from slave.processes.publisher.profiling_publisher import ProfilingPublisher
from slave.processes.stream.frame_reader import FrameReader
from slave.processes.video.video_manager import VideoManager
from slave.processes.worker.models.number_plate.nb.number_plate_nb_detector import NumberPlateNbDetector
from slave.profiling.profiler import Profiler
import time
from slave.processes.worker.models.number_plate.plate_enricher import PlateEnricher
from slave.processes.worker.worker_factory import WorkerIdentity
from slave.runtimes.common.round_robin_scheduler import SchedulerBackedWorker
from slave.utils.time_utils import ist_now_iso


class NumberPlateNbWorker(SchedulerBackedWorker):
    def __init__(
        self,
        job: JobConfig,
        assignment: ModelAssignment,
        worker_key: str,
        model_config: dict,
        models_dir: str,
        detection_publisher: DetectionPublisher,
        frame_uploader: FrameUploader,
        video_manager: VideoManager,
        identity: WorkerIdentity,
        profiling_publisher: ProfilingPublisher,
    ) -> None:
        self._job = job
        self._worker_key = worker_key
        self._detection_publisher = detection_publisher
        self._frame_uploader = frame_uploader
        self._video_manager = video_manager
        self._identity = identity
        self._profiling_publisher = profiling_publisher
        self._log = get_logger(__name__, job_id=job.job_id)
        self._profiler = Profiler(model_name=assignment.model_name)

        nb_cfg = model_config.get("nb", {})
        library_file = nb_cfg.get("library_file", "")
        model_path = os.path.join(models_dir, assignment.model_name)
        library_path = os.path.join(models_dir, library_file) if library_file else None
        self._enricher = PlateEnricher(model_config)

        url = job.camera_stream
        # Match the .pt path's FrameReader (always decode_every_n_frames=1)
        # -- previously defaulted to 3, giving .nb a third as many chances
        # to catch a passing plate as .pt got.
        self._frame_reader = FrameReader(
            worker_key, url, decode_every_n_frames=max(1, int(nb_cfg.get("decode_every_n_frames", 1)))
        )

        self._attach_scheduler(
            model_path,
            build_detector=lambda: NumberPlateNbDetector(
                model_path, library_path, model_config, assignment.model_name
            ),
            target_fps=float(nb_cfg.get("target_fps", 12.0)),
            process_interval=float(nb_cfg.get("process_interval", 0.0)),
        )

    def _process_frame(self, frame, raw_detections) -> None:
        # preprocessing – nothing heavy before inference in NB worker
        self._profiler.mark('preprocess')
        plate_detections = self._enricher.enrich(
            frame.image,
            raw_detections,
            frame.frame_id,
            frame_timestamp=frame.timestamp,
        )

        readable_plates = self._enricher.take_publishable()
        # profiling: inference already happened in scheduler (raw_detections)
        self._profiler.mark('inference')
        # queue wait measurement
        if hasattr(frame, 'timestamp'):
            self._profiler.record_queue_wait((time.time() - frame.timestamp) * 1000.0)
        # post‑process (enrichment) – now after enrichment work done
        self._profiler.mark('postprocess')
        # finish and publish profiling report
        self._profiler.stop()
        try:
            self._profiling_publisher.publish(self._profiler.report())
        except Exception:
            self._log.exception('failed to publish profiling data')

        object_key = url = None
        annotated = None
        if plate_detections:
            annotated = draw_detections(frame.image, plate_detections)
        if readable_plates:
            upload_info = self._frame_uploader.upload_frame(self._worker_key, frame.frame_id, annotated)
            if upload_info is not None:
                object_key, url = upload_info

        width = height = None
        if object_key is not None:
            height, width = (int(v) for v in frame.image.shape[:2])

        telemetry = group_detections(readable_plates, self._job, tracking_id=object_key)

        result = DetectionResult(
            job_id=self._job.job_id,
            frame_id=frame.frame_id,
            occurred_at=ist_now_iso(),
            device_name=self._identity.device_name,
            branch_id=self._job.branch_id,
            branch_name=self._job.branch_name,
            device_id=self._identity.device_id,
            zone_id=self._job.zone_id,
            zone_name=self._job.zone_name,
            camera_id=self._job.camera_id,
            camera_name=self._job.camera_name,
            camera_stream=self._job.camera_stream,
            camera_status=self._frame_reader.get_status(),
            latitude=self._job.latitude,
            longitude=self._job.longitude,
            telemetry=telemetry,
            object_key=object_key,
            url=url,
            width=width,
            height=height,
        )
        self._detection_publisher.publish_detection(result)

        if plate_detections:
            self._video_manager.feed_frame(annotated)

    def _cleanup(self) -> None:
        try:
            self._frame_reader.stop()
        except Exception:
            self._log.exception("error stopping frame reader")
        # NOTE: the detector/NPU handle is NOT released here -- it is
        # shared across every job using this model_path.
        # scheduler_registry.unregister_job() (called above, before
        # _cleanup) already releases it, but only once this was the
        # last job using that model_path.
        try:
            self._enricher.release()
        except Exception:
            self._log.exception("error releasing plate enricher")
        try:
            self._video_manager.close()
        except Exception:
            self._log.exception("error closing video manager")
