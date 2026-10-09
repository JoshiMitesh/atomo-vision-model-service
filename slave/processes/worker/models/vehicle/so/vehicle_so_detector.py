"""Vehicle `.so` detector — native shared-library inference path.

Same letterbox/DFL/NMS pipeline as vehicle_nb_detector.py (including
its placeholder-decode caveat — see that module's docstring), but
inference goes through runtimes/npu/so_runtime.py (a directly-loaded
native `.so` inference engine, not a KSNN/ASNN `.nb` model). Inherits
the same "not yet functional" status as runtimes/npu/so_runtime.py
today (SoRuntime._init()/infer() raise NotImplementedError until the
board's real .so ABI is wired up) — true for every other model's .so
path too, not a new gap introduced by Vehicle.

There is no standalone `.so`-only vehicle checkpoint distinct from the
`.nb` companion library expected in this folder (see
processes/worker/models/vehicle/ — vehicle.pt/vehicle.nb/vehicle.so);
a `model_extension: .so` job loads vehicle.so directly through this
path once the board ABI is implemented.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from slave.models.detection import Detection
from slave.runtimes.common import runtime_utils as ru
from slave.runtimes.npu.so_runtime import SoRuntime

CLASSES = ("Helmet", "bus", "car", "motorcycle", "no-seatbelt", "no_helmet", "other", "seatbelt", "truck")
NUM_CLS = len(CLASSES)
REG_MAX = 16
LISTSIZE = NUM_CLS + 4 * REG_MAX  # 73


class VehicleSoDetector:
    def __init__(self, library_path: str, model_config: dict, model_name: str) -> None:
        self._model_name = model_name
        self._display_name = model_config.get("display_name") or model_name
        thresholds = model_config.get("thresholds", {})
        # Same decode math as vehicle_nb_detector.py, so it reads the
        # same nb: block for the extra geometry-filter knobs rather
        # than duplicating them under a separate so: section.
        nb_cfg = model_config.get("nb", {})
        self._obj_thresh = float(nb_cfg.get("obj_thresh", thresholds.get("obj_thresh", 0.25)))
        self._nms_thresh = float(nb_cfg.get("nms_thresh", thresholds.get("nms_thresh", 0.45)))
        self._dfl_certainty_thresh = float(nb_cfg.get("dfl_certainty_thresh", 0.05))
        self._min_aspect_ratio = float(nb_cfg.get("min_aspect_ratio", 0.20))
        self._max_aspect_ratio = float(nb_cfg.get("max_aspect_ratio", 5.0))
        self._min_box_size_ratio = float(nb_cfg.get("min_box_size_ratio", 0.10))
        self._max_detections = int(nb_cfg.get("max_detections", 100))
        self._ignore_classes = frozenset(model_config.get("ignore_classes", ()))

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
            tensor = ru.reshape_npu_output(raw, LISTSIZE)[0]
            grid_size = tensor.shape[-1]
            stride = self._input_size / grid_size
            boxes, scores, class_ids = self._decode_grid(tensor, stride)
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
        if len(keep) > self._max_detections:
            keep = sorted(keep, key=lambda i: scores[i], reverse=True)[: self._max_detections]

        boxes = ru.unletterbox_boxes(boxes[keep], info)
        scores = scores[keep]
        class_ids = class_ids[keep]

        detections: List[Detection] = []
        for box, score, cid in zip(boxes, scores, class_ids):
            class_name = CLASSES[int(cid)]
            if class_name in self._ignore_classes:
                continue
            detections.append(
                Detection(
                    class_id=int(cid),
                    class_name=class_name,
                    confidence=float(score),
                    bbox=tuple(box.tolist()),
                    model_name=self._display_name,
                )
            )
        return detections

    def _decode_grid(self, tensor: np.ndarray, stride: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Same decode as vehicle_nb_detector.py's _decode_grid() — see
        that module's docstring for the full rationale.
        """
        _, grid_h, grid_w = tensor.shape
        flat = tensor.reshape(tensor.shape[0], -1)

        cls_probs = ru.sigmoid(flat[:NUM_CLS])
        class_ids = np.argmax(cls_probs, axis=0)
        scores = cls_probs[class_ids, np.arange(cls_probs.shape[1])]

        keep = scores >= self._obj_thresh
        if not np.any(keep):
            return self._empty()

        dfl = flat[NUM_CLS:, keep].reshape(4, REG_MAX, -1)
        dfl = ru.softmax(dfl, axis=1)

        certainty = dfl.max(axis=1).mean(axis=0)
        cert_keep = certainty >= self._dfl_certainty_thresh
        if not np.any(cert_keep):
            return self._empty()

        idx = np.nonzero(keep)[0][cert_keep]
        dfl = dfl[:, :, cert_keep]
        cand_scores = scores[keep][cert_keep]
        cand_class_ids = class_ids[keep][cert_keep]

        bins = np.arange(REG_MAX, dtype=np.float32)
        ltrb = np.tensordot(bins, dfl, axes=(0, 1)).T * stride

        rows, cols = np.divmod(idx, grid_w)
        cx = (cols + 0.5) * stride
        cy = (rows + 0.5) * stride
        x1 = cx - ltrb[:, 0]
        y1 = cy - ltrb[:, 1]
        x2 = cx + ltrb[:, 2]
        y2 = cy + ltrb[:, 3]

        w = x2 - x1
        h = y2 - y1
        min_size = stride * self._min_box_size_ratio
        aspect = w / (h + 1e-6)
        geom_keep = (
            (w >= min_size)
            & (h >= min_size)
            & (aspect >= self._min_aspect_ratio)
            & (aspect <= self._max_aspect_ratio)
        )
        if not np.any(geom_keep):
            return self._empty()

        boxes = np.stack([x1, y1, x2, y2], axis=1)[geom_keep].astype(np.float32)
        return (
            boxes,
            cand_scores[geom_keep].astype(np.float32),
            cand_class_ids[geom_keep].astype(np.int64),
        )

    @staticmethod
    def _empty() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    def release(self) -> None:
        self._runtime.release()
