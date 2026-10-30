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

    def infer(self, frame: np.ndarray, conf: float, iou: float, imgsz: int | None = None) -> Any:
        predict_kwargs = dict(
            source=frame,
            conf=conf,
            iou=iou,
            device=self._device,
            verbose=False,
        )
        if imgsz is not None:
            predict_kwargs["imgsz"] = imgsz
        # Only pass `half` when actually requesting it — some Ultralytics
        # versions emit a "deprecated, use quantize" warning on *every*
        # predict() call whenever this kwarg is present at all, even
        # half=False, which floods the log at inference framerate.
        if self._half:
            predict_kwargs["half"] = True
        return self._model.predict(**predict_kwargs)[0]

    def release(self) -> None:
        self._model = None
