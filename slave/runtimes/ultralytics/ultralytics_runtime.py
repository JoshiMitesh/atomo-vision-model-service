"""Thin adapter around the Ultralytics/PyTorch runtime.

Isolates the third-party API surface so person_pt_detector.py never
imports `ultralytics` directly (mirrors the isolation CLAUDE.md
requires for the NPU runtimes).
"""

from __future__ import annotations

from typing import Any

import numpy as np


class UltralyticsRuntime:
    def __init__(self, model_path: str, device: str = "auto", half: bool = False) -> None:
        from ultralytics import YOLO  # imported lazily so .nb/.so-only deployments don't need torch

        self._model = YOLO(model_path)
        self._device = None if device == "auto" else device
        self._half = half

    def infer(self, frame: np.ndarray, conf: float, iou: float) -> Any:
        return self._model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            device=self._device,
            half=self._half,
            verbose=False,
        )[0]

    def release(self) -> None:
        self._model = None
