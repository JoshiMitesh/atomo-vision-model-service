"""Profiling utilities for model inference.

Collects CPU, memory, system load, thread usage, NPU utilisation, and
pipeline stage timing. The public API remains ``start()``, ``mark(stage)``
and ``stop()`` plus ``report()`` which returns a dict ready for NATS
publishing.
"""

import time
import os
from .gpu_util import read_gpu_stats
from datetime import datetime, timezone
from typing import Dict, Any
import psutil
from .galcore_util import read_npu_load


class Profiler:
    """Collect telemetry for a single inference lifecycle.

    Example usage::

        profiler = Profiler(model_name="person.pt")
        profiler.start()
        # preprocessing...
        profiler.mark("preprocess")
        # inference...
        profiler.mark("inference")
        # post‑processing...
        profiler.mark("postprocess")
        profiler.stop()
        payload = profiler.report()

    The ``report`` dict contains all fields required by the master
    subscriber, plus the additional NPU and stage timing data.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._proc = psutil.Process()
        self._start_time: float | None = None
        self._end_time: float | None = None
        self._stage_times: Dict[str, float] = {}
        self._cpu_percent: float | None = None
        self._mem_info: psutil._common.smem | None = None
        # ---------- new fields ----------
        self._queue_lengths: Dict[str, int] = {}
        self._frame_timestamps: Dict[int, float] = {}
        self._camera_decode_ms: float = 0.0
        self._queue_wait_ms: float = 0.0

    # -----------------------------------------------------------------
    # Lifecycle methods
    # -----------------------------------------------------------------
    def start(self) -> None:
        """Mark the beginning of the inference operation.

        Captures a baseline CPU percent (non‑blocking) and a snapshot of
        virtual memory.
        """
        self._start_time = time.time()
        self._cpu_percent = psutil.cpu_percent(interval=None)
        self._mem_info = psutil.virtual_memory()

    # -----------------------------------------------------------------
    # New instrumentation helpers
    # -----------------------------------------------------------------
    def record_queue_len(self, name: str, length: int) -> None:
        self._queue_lengths[name] = length

    def record_frame_capture(self, frame_id: int, ts: float) -> None:
        self._frame_timestamps[frame_id] = ts

    def record_camera_decode(self, elapsed_ms: float) -> None:
        self._camera_decode_ms = elapsed_ms

    def record_queue_wait(self, elapsed_ms: float) -> None:
        self._queue_wait_ms = elapsed_ms

    def mark(self, stage: str) -> None:
        """Record a timestamp for a named pipeline stage (e.g. ``preprocess``)."""
        self._stage_times[stage] = time.time()

    def stop(self) -> None:
        """Mark the end of the inference operation."""
        self._end_time = time.time()

    # -----------------------------------------------------------------
    # Timing helpers
    # -----------------------------------------------------------------
    def _elapsed_ms(self, start_key: str, end_key: str) -> float:
        """Return the elapsed time between two marked stages in ms."""
        start = self._stage_times.get(start_key)
        end = self._stage_times.get(end_key)
        if start is None or end is None:
            return 0.0
        return (end - start) * 1000.0

    # -----------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------
    def report(self) -> Dict[str, Any]:
        """Build the enriched payload.

        All numeric values are rounded to two decimal places. Missing optional
        values are emitted as ``None`` (JSON null).
        """
        # Process / OS metrics ------------------------------------------------
        cpu_total = self._cpu_percent if self._cpu_percent is not None else psutil.cpu_percent(interval=None)
        cpu_per_core = psutil.cpu_percent(percpu=True, interval=None)
        system_load = os.getloadavg()               # 1‑,5‑,15‑min load averages
        ctx_sw = psutil.cpu_stats().ctx_switches
        thread_cpu = {
            f"tid_{t.id}": round(t.user_time + t.system_time, 2)
            for t in self._proc.threads()
        }

        mem = self._proc.memory_info()
        rss_mb = mem.rss / (1024 * 1024)
        pss_mb = getattr(mem, "pss", rss_mb) / (1024 * 1024)
        swap_mb = psutil.swap_memory().used / (1024 * 1024)

        # NPU utilisation -----------------------------------------------------
        npu_util = read_npu_load()

        # Stage timing ----------------------------------------------------------
        preprocess_ms = self._elapsed_ms("start", "preprocess")
        inference_ms = self._elapsed_ms("preprocess", "inference")
        postprocess_ms = self._elapsed_ms("inference", "postprocess")
        total_ms = 0.0
        if self._start_time is not None and self._end_time is not None:
            total_ms = (self._end_time - self._start_time) * 1000.0

        gpu_stats = read_gpu_stats()
        payload: Dict[str, Any] = {
            "model_name": self.model_name,
            "cpu_percent": round(cpu_total, 2),
            "cpu_per_core": {str(i): round(v, 2) for i, v in enumerate(cpu_per_core)},
            "system_load": [round(v, 2) for v in system_load],
            "context_switches": ctx_sw,
            "thread_cpu": thread_cpu,
            "memory_total_mb": round(psutil.virtual_memory().total / (1024 * 1024), 2),
            "memory_used_mb": round(psutil.virtual_memory().used / (1024 * 1024), 2),
            "memory_rss_mb": round(rss_mb, 2),
            "memory_pss_mb": round(pss_mb, 2),
            "swap_used_mb": round(swap_mb, 2),
            "gpu_utilization": round(gpu_stats.get("gpu_utilization", 0), 2) if gpu_stats.get("gpu_utilization") is not None else None,
            "gpu_memory_total_mb": round(gpu_stats.get("gpu_memory_total_mb", 0), 2) if gpu_stats.get("gpu_memory_total_mb") is not None else None,
            "gpu_memory_used_mb": round(gpu_stats.get("gpu_memory_used_mb", 0), 2) if gpu_stats.get("gpu_memory_used_mb") is not None else None,
            "gpu_temperature_c": round(gpu_stats.get("gpu_temperature_c", 0), 2) if gpu_stats.get("gpu_temperature_c") is not None else None,
            "npu_utilization": round(npu_util, 2) if npu_util is not None else None,
            "preprocess_ms": round(preprocess_ms, 2),
            "inference_ms": round(inference_ms, 2),
            "postprocess_ms": round(postprocess_ms, 2),
            "camera_decode_ms": round(self._camera_decode_ms, 2),
            "queue_wait_ms": round(self._queue_wait_ms, 2),
            "queue_lengths": self._queue_lengths,
            "duration_ms": round(total_ms, 2),
            "end_to_end_latency_ms": (
                round((time.time() - max(self._frame_timestamps.values())) * 1000.0, 2)
                if self._frame_timestamps else None
            ),
            "timestamp": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        }
        return payload

