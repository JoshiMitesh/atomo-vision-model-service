"""Reference-counted, per-model detector sharing across camera jobs.

Each camera keeps its own RTSP reader, tracker, OCR state, and publishing
flow. Identical number-plate model jobs share one loaded detector and
serialize only detector inference; OCR and event handling remain outside
the detector lock. This avoids loading multiple copies of the same model
and prevents concurrent calls into a single NPU/runtime handle.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Hashable


@dataclass
class _Entry:
    detector: Any
    inference_lock: threading.Lock
    references: int = 0


_registry_lock = threading.Lock()
_registry: dict[Hashable, _Entry] = {}


class SharedDetectorLease:
    """A job-scoped lease over a process-shared detector."""

    def __init__(self, key: Hashable, entry: _Entry) -> None:
        self._key = key
        self._entry = entry
        self._released = False
        self._lease_lock = threading.Lock()

    def detect(self, frame):
        with self._lease_lock:
            if self._released:
                raise RuntimeError("detector lease has been released")
        with self._entry.inference_lock:
            return self._entry.detector.detect(frame)

    def release(self) -> None:
        global _registry
        with self._lease_lock:
            if self._released:
                return
            self._released = True

        detector_to_release = None
        with _registry_lock:
            entry = _registry.get(self._key)
            if entry is not self._entry:
                return
            entry.references -= 1
            if entry.references <= 0:
                _registry.pop(self._key, None)
                detector_to_release = entry.detector

        if detector_to_release is not None:
            # Wait for any in-flight inference before releasing the runtime.
            with self._entry.inference_lock:
                try:
                    detector_to_release.release()
                except Exception:
                    # Worker shutdown should not fail because a backend's
                    # optional release hook raises.
                    pass


def acquire_shared_detector(key: Hashable, factory: Callable[[], Any]) -> SharedDetectorLease:
    """Acquire or lazily create a detector for a stable model/backend key."""
    with _registry_lock:
        entry = _registry.get(key)
        if entry is None:
            entry = _Entry(detector=factory(), inference_lock=threading.Lock())
            _registry[key] = entry
        entry.references += 1
        return SharedDetectorLease(key, entry)
