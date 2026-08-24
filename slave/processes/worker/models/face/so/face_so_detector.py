"""Face `.so` detector — native shared-library inference path.

Mirrors person_so_detector.py's plumbing exactly. Inherits the same
"not yet functional" status as runtimes/npu/so_runtime.py today
(SoRuntime._init()/infer() raise NotImplementedError until the board's
real .so ABI is wired up) — true for the Person .so path too, not a
new gap introduced by Face.

IMPORTANT: like face_nb_detector.py, this is NOT a DFL (reg_max=16)
model — see that file's docstring for the confirmed LISTSIZE=5 direct
box-regression format this face model actually uses.
"""

from __future__ import annotations

from typing import List

import numpy as np

from slave.models.detection import Detection
from slave.runtimes.common import runtime_utils as ru
from slave.runtimes.npu.so_runtime import SoRuntime

NUM_CLS = 1
CLASSES = ("face",)
CLASS_ID = 1
LISTSIZE = NUM_CLS + 4  # 5 — class score + direct l/t/r/b box offsets, no DFL


class FaceSoDetector:
    def __init__(self, library_path: str, model_config: dict, model_name: str) -> None:
        self._model_name = model_name
        thresholds = model_config.get("thresholds", {})
        self._obj_thresh = float(thresholds.get("obj_thresh", 0.35))
        self._nms_thresh = float(thresholds.get("nms_thresh", 0.45))

        input_cfg = model_config.get("input", {})
        keyword = str(input_cfg.get("large_size_keyword", "1024"))
        self._input_size = int(
            input_cfg.get("large_size", 1024) if keyword in model_name else input_cfg.get("default_size", 640)
        )
        self._pad_value = int(input_cfg.get("pad_value", 114))

        so_cfg = model_config.get("so", {})
        self._runtime = SoRuntime(library_path=library_path, device_id=int(so_cfg.get("device_id", 0)))

    def detect(self, frame: np.ndarray) -> List[Detection]:
        letterboxed, info = ru.letterbox(frame, self._input_size, self._pad_value)
        input_tensor = ru.to_input_tensor(letterboxed)

        raw_outputs = self._runtime.infer(input_tensor)
        outputs = sorted(raw_outputs, key=lambda t: t.size)

        all_boxes, all_scores = [], []
        for tensor in outputs:
            tensor = ru.reshape_npu_output(tensor, LISTSIZE)
            grid_size = tensor.shape[-1]
            stride = self._input_size // grid_size
            boxes, scores = ru.decode_grid_direct(tensor, NUM_CLS, stride, self._obj_thresh)
            if boxes.shape[0]:
                all_boxes.append(boxes)
                all_scores.append(scores)

        if not all_boxes:
            return []

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        keep = ru.nms(boxes, scores, self._nms_thresh)
        boxes = ru.unletterbox_boxes(boxes[keep], info)
        scores = scores[keep]

        return [
            Detection(
                class_id=CLASS_ID,
                class_name=CLASSES[0],
                confidence=float(score),
                bbox=tuple(box.tolist()),
                model_name=self._model_name,
            )
            for box, score in zip(boxes, scores)
        ]

    def release(self) -> None:
        self._runtime.release()
