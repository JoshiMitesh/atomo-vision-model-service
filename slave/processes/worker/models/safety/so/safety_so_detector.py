"""Safety `.so` detector — native shared-library inference path.

Same letterbox/DFL/NMS pipeline as the `.nb` path, but inference goes
through runtimes/npu/so_runtime.py (a directly-loaded native `.so`
inference engine, not a KSNN/ASNN `.nb` model). Inherits the same
"not yet functional" status as runtimes/npu/so_runtime.py today
(SoRuntime._init()/infer() raise NotImplementedError until the
board's real .so ABI is wired up) — true for Person/Face's .so paths
too, not a new gap introduced by Safety.
"""

from __future__ import annotations

from typing import List

import numpy as np

from slave.models.detection import Detection
from slave.runtimes.common import runtime_utils as ru
from slave.runtimes.npu.so_runtime import SoRuntime

CLASSES = (
    "boots",
    "gloves",
    "hardhat",
    "no_boots",
    "no_gloves",
    "no_hardhat",
    "no_vest",
    "person",
    "vest",
)
NUM_CLS = len(CLASSES)
REG_MAX = 16
LISTSIZE = NUM_CLS + 4 * REG_MAX  # 73


class SafetySoDetector:
    def __init__(self, library_path: str, model_config: dict, model_name: str) -> None:
        self._model_name = model_name
        self._display_name = model_config.get("display_name") or model_name
        thresholds = model_config.get("thresholds", {})
        self._obj_thresh = float(thresholds.get("obj_thresh", 0.25))
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
        outputs = sorted(raw_outputs, key=lambda t: t.size, reverse=True)

        all_boxes, all_scores, all_class_ids = [], [], []
        for raw in outputs:
            tensor = ru.reshape_npu_output(raw, LISTSIZE)
            grid_size = tensor.shape[-1]
            stride = self._input_size // grid_size
            boxes, scores, class_ids = ru.decode_grid_multi(tensor, NUM_CLS, REG_MAX, stride, self._obj_thresh)
            if boxes.shape[0]:
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_class_ids.append(class_ids)

        if not all_boxes:
            return []

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        class_ids = np.concatenate(all_class_ids, axis=0)

        keep = ru.nms_multiclass(boxes, scores, class_ids, self._nms_thresh)
        if not keep:
            return []
        boxes = ru.unletterbox_boxes(boxes[keep], info)
        scores = scores[keep]
        class_ids = class_ids[keep]

        return [
            Detection(
                class_id=int(cid),
                class_name=CLASSES[int(cid)],
                confidence=float(score),
                bbox=tuple(box.tolist()),
                model_name=self._display_name,
            )
            for box, score, cid in zip(boxes, scores, class_ids)
        ]

    def release(self) -> None:
        self._runtime.release()
