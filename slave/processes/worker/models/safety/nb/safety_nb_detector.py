"""Safety `.nb` detector — Amlogic A311D NPU inference path.

Mirrors person_nb_detector.py's runtime plumbing (letterbox ->
NbRuntime -> decode -> NMS -> unletterbox), but deliberately WITHOUT
Person's hand-tuned aspect-ratio / small-box-area / head-texture
geometry filters — those are person-shape-specific and would be wrong
for hardhats/boots/gloves/vests (CLAUDE.md §7-§8: pt/nb/so pipelines
may differ per model; only the final Detection contract is shared).

Unlike Person/Face (both single-class), Safety is a 9-class DFL model
(confirmed via the .pt checkpoint: reg_max=16, nc=9, no=73) — this
uses runtime_utils.decode_grid_multi()/nms_multiclass() instead of
decode_grid()/nms(), so each box keeps its own class_id instead of
being collapsed to one fixed class.

NOTE: safety.nb's real compiled input resolution has NOT been
confirmed on-device yet (see config/models/safety.yaml's input:
comment) — verify before trusting default_size here.
"""

from __future__ import annotations

from typing import List

import numpy as np

from slave.models.detection import Detection
from slave.runtimes.common import runtime_utils as ru
from slave.runtimes.npu.nb_runtime import NbRuntime

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


class SafetyNbDetector:
    def __init__(self, model_path: str, library_path: str, model_config: dict, model_name: str) -> None:
        self._model_name = model_name
        self._display_name = model_config.get("display_name") or model_name

        nb_cfg = model_config.get("nb", {})
        thresholds = model_config.get("thresholds", {})
        self._obj_thresh = float(nb_cfg.get("obj_thresh", thresholds.get("obj_thresh", 0.25)))
        self._nms_thresh = float(nb_cfg.get("nms_thresh", thresholds.get("nms_thresh", 0.45)))

        input_cfg = model_config.get("input", {})
        keyword = str(input_cfg.get("large_size_keyword", "1024"))
        self._input_size = int(
            input_cfg.get("large_size", 1024) if keyword in model_name else input_cfg.get("default_size", 640)
        )
        self._pad_value = int(input_cfg.get("pad_value", 114))
        self._lb_canvas = np.full((self._input_size, self._input_size, 3), self._pad_value, dtype=np.uint8)

        self._runtime = NbRuntime(
            model_path=model_path,
            library_path=library_path,
            level=int(nb_cfg.get("level", 0)),
            device_id=int(nb_cfg.get("device_id", 0)),
            board=str(nb_cfg.get("board", "Electron")),
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        letterboxed, info = ru.letterbox(frame, self._input_size, self._pad_value, canvas=self._lb_canvas)
        input_tensor = ru.to_input_tensor(letterboxed)

        raw_outputs = self._runtime.infer(input_tensor)
        # Largest tensor first (finest grid / smallest stride), same
        # convention as person_nb_detector.py.
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
