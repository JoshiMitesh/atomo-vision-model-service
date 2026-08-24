"""Face `.nb` detector — Amlogic A311D NPU inference path.

Mirrors person_nb_detector.py's runtime plumbing (letterbox ->
NbRuntime -> decode -> NMS -> unletterbox), but deliberately without
Person's hand-tuned aspect-ratio / small-box-area / head-texture
geometry filters — those are person-shape-specific (CLAUDE.md §7-§8:
pt/nb/so pipelines may differ per model; only the final Detection
contract is shared).

IMPORTANT: unlike person.nb, this face.nb is NOT a DFL (reg_max=16)
model — it was compiled from an Ultralytics `model.export(...,
nms=False)` with a direct 1-class + 4-box-offset head (confirmed via
this exact model's ONNX export: 3 outputs of shape [1,5,80,80] /
[1,5,40,40] / [1,5,20,20] — i.e. LISTSIZE=5, not 65). Using the DFL
decode here fails outright (output size not divisible by 65); this
uses runtime_utils.decode_grid_direct() instead, matching a verified
working reference decode for this exact model
(face_attrs_npu_rtsp.py's decode_scale()).
"""

from __future__ import annotations

from typing import List

import numpy as np

from slave.models.detection import Detection
from slave.runtimes.common import runtime_utils as ru
from slave.runtimes.npu.nb_runtime import NbRuntime

NUM_CLS = 1
CLASSES = ("face",)
CLASS_ID = 1
LISTSIZE = NUM_CLS + 4  # 5 — class score + direct l/t/r/b box offsets, no DFL


class FaceNbDetector:
    def __init__(self, model_path: str, library_path: str, model_config: dict, model_name: str) -> None:
        self._model_name = model_name
        nb_cfg = model_config.get("nb", {})
        thresholds = model_config.get("thresholds", {})
        self._obj_thresh = float(nb_cfg.get("obj_thresh", thresholds.get("obj_thresh", 0.35)))
        self._nms_thresh = float(nb_cfg.get("nms_thresh", thresholds.get("nms_thresh", 0.45)))

        input_cfg = model_config.get("input", {})
        keyword = str(input_cfg.get("large_size_keyword", "1024"))
        self._input_size = int(
            input_cfg.get("large_size", 1024) if keyword in model_name else input_cfg.get("default_size", 640)
        )
        self._pad_value = int(input_cfg.get("pad_value", 114))

        self._runtime = NbRuntime(
            model_path=model_path,
            library_path=library_path,
            level=int(nb_cfg.get("level", 0)),
            device_id=int(nb_cfg.get("device_id", 0)),
            board=str(nb_cfg.get("board", "Electron")),
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        letterboxed, info = ru.letterbox(frame, self._input_size, self._pad_value)
        input_tensor = ru.to_input_tensor(letterboxed)

        raw_outputs = self._runtime.infer(input_tensor)
        # Largest tensor first (finest grid / smallest stride), same
        # convention as person_nb_detector.py.
        outputs = sorted(raw_outputs, key=lambda t: t.size, reverse=True)

        all_boxes, all_scores = [], []
        for raw in outputs:
            tensor = ru.reshape_npu_output(raw, LISTSIZE)
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
        if not keep:
            return []
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
