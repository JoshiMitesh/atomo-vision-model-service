"""Fire `.nb` detector — Amlogic A311D NPU inference path.

Preprocessing (letterbox), inference (via runtimes/npu/nb_runtime.py),
DFL decoding, confidence filtering, per-class NMS, and coordinate
conversion — all Fire-model-specific, per CLAUDE.md §11-§13.

Ported from a confirmed-working low-CPU 3-class fire/smoke NPU
reference script (YOLOv8-DFL head, classes-first channel layout,
sigmoid class activation, reg_max=16 — matches fire.pt's Detect head:
nc=3, no=67). Its post-processing is: one confidence gate on the
per-cell max-class score, a DFL-distribution certainty filter, a
tiny-box size filter, and a wide aspect-ratio filter (fire/smoke
shapes are far less regular than a person's silhouette, hence
0.10-6.0 instead of Person's "only tall boxes" filter), then per-class
NMS. Self-contained here rather than folded into runtime_utils.py,
same rationale as person_nb_detector.py (CLAUDE.md §7-§8: pt/nb/so
pipelines may differ per model; only the final Detection contract is
shared).

Preprocessing note: unlike person_nb_detector.py (which converts
BGR->RGB via runtime_utils.to_input_tensor before inference), the
fire reference explicitly feeds BGR pixels
(`cv2.dnn.blobFromImage(..., swapRB=False)`) and relies solely on
NbRuntime's `reorder='2 1 0'` nn_inference argument to reorder
channels at the NPU. Stacking that reorder argument on top of an
extra manual BGR->RGB flip would double-flip color order, so this
detector uses its own _to_input_tensor_bgr() instead of
runtime_utils.to_input_tensor().

"other" is a hard-negative/distractor class from the stage-2 fine-tune
(see config/models/fire.yaml's `classes:`/`ignore_classes:` comments)
— decoded and NMS'd like any other class (so it can still suppress
overlapping fire/smoke false positives), then dropped from the
returned detections.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from slave.models.detection import Detection
from slave.runtimes.common import runtime_utils as ru
from slave.runtimes.npu.nb_runtime import NbRuntime

CLASSES = ("fire", "other", "smoke")
NUM_CLS = len(CLASSES)
REG_MAX = 16
LISTSIZE = NUM_CLS + 4 * REG_MAX  # 67


def _to_input_tensor_bgr(letterboxed_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 HWC -> BGR float32 CHW, normalized to [0, 1] — no RGB
    conversion (see module docstring)."""
    chw = letterboxed_bgr.astype(np.float32).transpose(2, 0, 1) / 255.0
    return np.ascontiguousarray(chw)


class FireNbDetector:
    def __init__(self, model_path: str, library_path: str, model_config: dict, model_name: str) -> None:
        self._model_name = model_name
        self._display_name = model_config.get("display_name") or model_name

        nb_cfg = model_config.get("nb", {})
        thresholds = model_config.get("thresholds", {})
        # NB path uses its own tuned thresholds (config/models/fire.yaml
        # nb: block) — falls back to the shared thresholds: block only if
        # not overridden there, so .pt keeps its own defaults.
        self._obj_thresh = float(nb_cfg.get("obj_thresh", thresholds.get("obj_thresh", 0.15)))
        self._nms_thresh = float(nb_cfg.get("nms_thresh", thresholds.get("nms_thresh", 0.45)))
        self._dfl_certainty_thresh = float(nb_cfg.get("dfl_certainty_thresh", 0.05))
        self._min_aspect_ratio = float(nb_cfg.get("min_aspect_ratio", 0.10))
        self._max_aspect_ratio = float(nb_cfg.get("max_aspect_ratio", 6.0))
        self._min_box_size_ratio = float(nb_cfg.get("min_box_size_ratio", 0.10))  # fraction of stride
        self._max_detections = int(nb_cfg.get("max_detections", 100))
        self._ignore_classes = frozenset(model_config.get("ignore_classes", ()))

        input_cfg = model_config.get("input", {})
        keyword = str(input_cfg.get("large_size_keyword", "1024"))
        self._input_size = int(
            input_cfg.get("large_size", 1024) if keyword in model_name else input_cfg.get("default_size", 1024)
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
        input_tensor = _to_input_tensor_bgr(letterboxed)

        raw_outputs = self._runtime.infer(input_tensor)
        # Largest tensor first (finest grid / smallest stride), same
        # convention as person_nb_detector.py / safety_nb_detector.py.
        outputs = sorted(raw_outputs, key=lambda t: t.size, reverse=True)

        all_boxes, all_scores, all_class_ids = [], [], []
        for raw in outputs:
            tensor = ru.reshape_npu_output(raw, LISTSIZE)[0]  # drop the leading batch dim -> (LISTSIZE, H, W)
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
        # Reference caps the surviving candidate count, strongest first, as
        # a worst-case-CPU bound against an unusually noisy frame.
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
        """One output grid, shaped (LISTSIZE, grid_h, grid_w) -> filtered
        fire/smoke/other candidate boxes (letterboxed pixel space), their
        scores, and their class ids.

        Single confidence gate on the per-cell max-class score, then a
        DFL-distribution certainty filter, a tiny-box size filter, and a
        wide aspect-ratio filter — ported from the reference pipeline's
        process().
        """
        _, grid_h, grid_w = tensor.shape
        flat = tensor.reshape(tensor.shape[0], -1)  # (LISTSIZE, N)

        cls_probs = ru.sigmoid(flat[:NUM_CLS])  # (NUM_CLS, N)
        class_ids = np.argmax(cls_probs, axis=0)  # (N,)
        scores = cls_probs[class_ids, np.arange(cls_probs.shape[1])]  # (N,) — score of the argmax class

        keep = scores >= self._obj_thresh
        if not np.any(keep):
            return self._empty()

        dfl = flat[NUM_CLS:, keep].reshape(4, REG_MAX, -1)  # (4, REG_MAX, K)
        dfl = ru.softmax(dfl, axis=1)

        # Mean of each of the 4 DFL groups' peak probability — how
        # "certain" the box regression is, independent of class score.
        certainty = dfl.max(axis=1).mean(axis=0)  # (K,)
        cert_keep = certainty >= self._dfl_certainty_thresh
        if not np.any(cert_keep):
            return self._empty()

        idx = np.nonzero(keep)[0][cert_keep]
        dfl = dfl[:, :, cert_keep]
        cand_scores = scores[keep][cert_keep]
        cand_class_ids = class_ids[keep][cert_keep]

        bins = np.arange(REG_MAX, dtype=np.float32)
        ltrb = np.tensordot(bins, dfl, axes=(0, 1)).T * stride  # (K, 4) pixels, l/t/r/b

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
