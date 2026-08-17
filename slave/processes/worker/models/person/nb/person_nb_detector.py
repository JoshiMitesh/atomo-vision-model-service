"""Person `.nb` detector — Amlogic A311D NPU inference path.

Preprocessing (letterbox), inference (via runtimes/npu/nb_runtime.py),
DFL decoding, confidence filtering, NMS, and coordinate conversion —
all Person-model-specific, per CLAUDE.md §11-§13.
"""

from __future__ import annotations

from typing import List

import numpy as np

from slave.models.detection import Detection
from slave.runtimes.common import runtime_utils as ru
from slave.runtimes.npu.nb_runtime import NbRuntime

NUM_CLS = 1
CLASSES = ("person",)
REG_MAX = 16
LISTSIZE = NUM_CLS + 4 * REG_MAX  # 65


class PersonNbDetector:
    def __init__(self, model_path: str, library_path: str, model_config: dict, model_name: str) -> None:
        thresholds = model_config.get("thresholds", {})
        self._obj_thresh = float(thresholds.get("obj_thresh", 0.20))
        self._nms_thresh = float(thresholds.get("nms_thresh", 0.65))

        input_cfg = model_config.get("input", {})
        keyword = str(input_cfg.get("large_size_keyword", "1024"))
        self._input_size = int(
            input_cfg.get("large_size", 1024) if keyword in model_name else input_cfg.get("default_size", 640)
        )
        self._pad_value = int(input_cfg.get("pad_value", 114))

        nb_cfg = model_config.get("nb", {})
        self._runtime = NbRuntime(
            model_path=model_path,
            library_path=library_path,
            level=int(nb_cfg.get("level", 0)),
            device_id=int(nb_cfg.get("device_id", 0)),
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        letterboxed, info = ru.letterbox(frame, self._input_size, self._pad_value)
        input_tensor = ru.to_input_tensor(letterboxed)

        raw_outputs = self._runtime.infer(input_tensor)
        outputs = sorted(raw_outputs, key=lambda t: t.size)

        all_boxes, all_scores = [], []
        for tensor in outputs:
            grid_size = tensor.shape[-1]
            stride = self._input_size // grid_size
            boxes, scores = ru.decode_grid(tensor, NUM_CLS, REG_MAX, stride, self._obj_thresh)
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
            Detection(class_id=0, class_name=CLASSES[0], confidence=float(score), bbox=tuple(box.tolist()))
            for box, score in zip(boxes, scores)
        ]

    def release(self) -> None:
        self._runtime.release()
