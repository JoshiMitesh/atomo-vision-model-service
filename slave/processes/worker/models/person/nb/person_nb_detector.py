"""Person `.nb` detector — Amlogic A311D NPU inference path.

Preprocessing (letterbox), inference (via runtimes/npu/nb_runtime.py),
DFL decoding, confidence filtering, NMS, and coordinate conversion —
all Person-model-specific, per CLAUDE.md §11-§13.

Ported from the updated reference pipeline (ultra-low-CPU 30fps
person detector): a two-stage confidence gate (low-bar candidate
gather, then DFL-distribution certainty), a person aspect-ratio
geometry filter, a small-box area filter, OpenCV NMS, and a
head-texture false-positive filter on the unletterboxed boxes. This
is intentionally self-contained here rather than folded into
runtime_utils.py — that module is shared with person_so_detector.py,
and none of this NB-specific tuning should silently change the `.so`
path (CLAUDE.md §7-§8: pt/nb/so may have entirely different
pipelines; only the final Detection contract is shared).
"""

from __future__ import annotations

import time
from typing import List, Tuple

import cv2
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
        self._model_name = model_name
        self.inference_ms = 0.0

        nb_cfg = model_config.get("nb", {})
        thresholds = model_config.get("thresholds", {})

        # NB path uses its own tuned thresholds (config/models/person.yaml
        # nb: block) — falls back to the shared thresholds: block only if
        # not overridden there, so .pt/.so keep their own defaults.
        self._obj_thresh = float(nb_cfg.get("obj_thresh", thresholds.get("obj_thresh", 0.20)))
        self._nms_thresh = float(nb_cfg.get("nms_thresh", thresholds.get("nms_thresh", 0.65)))
        self._candidate_gather_thresh = float(nb_cfg.get("candidate_gather_thresh", 0.15))
        self._dfl_certainty_thresh = float(nb_cfg.get("dfl_certainty_thresh", 0.18))
        self._min_aspect_ratio = float(nb_cfg.get("min_aspect_ratio", 0.80))
        self._min_small_box_area = float(nb_cfg.get("min_small_box_area", 2500.0))
        self._small_box_score_thresh = float(nb_cfg.get("small_box_score_thresh", 0.43))
        self._high_conf_bypass = float(nb_cfg.get("high_conf_bypass", 0.55))
        self._head_texture_std_thresh = float(nb_cfg.get("head_texture_std_thresh", 7.0))
        self._head_texture_score_thresh = float(nb_cfg.get("head_texture_score_thresh", 0.52))

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

        start = time.monotonic()
        raw_outputs = self._runtime.infer(input_tensor)
        self.inference_ms = (time.monotonic() - start) * 1000.0

        # Largest tensor first (finest grid / smallest stride — matches
        # the reference's candidate-gathering order).
        outputs = sorted(raw_outputs, key=lambda t: t.size, reverse=True)

        all_boxes, all_scores = [], []
        for flat_tensor in outputs:
            grid_size = int(round((flat_tensor.size / LISTSIZE) ** 0.5))
            tensor = flat_tensor.reshape(LISTSIZE, grid_size, grid_size)
            boxes, scores = self._decode_grid(tensor)
            if boxes.shape[0]:
                all_boxes.append(boxes)
                all_scores.append(scores)

        if not all_boxes:
            return []

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)

        keep = self._nms(boxes, scores)
        if keep.size == 0:
            return []
        boxes, scores = boxes[keep], scores[keep]

        boxes = ru.unletterbox_boxes(boxes, info)
        boxes, scores = self._filter_head_texture(frame, boxes, scores)

        return [
            Detection(
                class_id=0,
                class_name=CLASSES[0],
                confidence=float(score),
                bbox=tuple(box.tolist()),
                model_name=self._model_name,
            )
            for box, score in zip(boxes, scores)
        ]

    def _decode_grid(self, tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """One output grid, shaped (LISTSIZE, grid_h, grid_w) -> filtered
        person candidate boxes (letterboxed pixel space) + scores.

        Two-stage confidence gate: a low `candidate_gather_thresh` bar
        gathers everything worth decoding, then `dfl_certainty_thresh`
        (average peak of the four DFL distributions) discards fuzzy
        background noise before the aspect-ratio/area geometry filter.
        """
        _, _, grid_w = tensor.shape  # square grid (letterboxed onto a square canvas)
        stride = self._input_size / grid_w

        scores_map = ru.sigmoid(tensor[0:1, :, :])[0]  # (grid_h, grid_w) — single-class model
        ys, xs = np.where(scores_map >= self._candidate_gather_thresh)
        if ys.size == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        dfl = tensor[1:, ys, xs].T  # (K, 4*REG_MAX)
        groups = [ru.softmax(dfl[:, i * REG_MAX : (i + 1) * REG_MAX], axis=-1) for i in range(4)]

        certainty = sum(g.max(axis=1) for g in groups) / 4.0
        keep = certainty >= self._dfl_certainty_thresh
        if not np.any(keep):
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        ys, xs = ys[keep], xs[keep]
        groups = [g[keep] for g in groups]
        cand_scores = scores_map[ys, xs]

        bins = np.arange(REG_MAX, dtype=np.float32)
        left, top, right, bottom = (g @ bins for g in groups)

        cx = (xs + 0.5) * stride
        cy = (ys + 0.5) * stride
        x1 = cx - left * stride
        y1 = cy - top * stride
        x2 = cx + right * stride
        y2 = cy + bottom * stride

        # Human aspect-ratio geometry filter (discards horizontal
        # table/sofa/floor clutter) + small-box area filter (discards
        # low-confidence bag/luggage-sized boxes).
        w = x2 - x1
        h = y2 - y1
        aspect_ratio = h / (w + 1e-5)
        area = w * h
        area_mask = ~((area < self._min_small_box_area) & (cand_scores < self._small_box_score_thresh))
        geom_mask = (aspect_ratio >= self._min_aspect_ratio) & area_mask

        if not np.any(geom_mask):
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        boxes = np.stack([x1, y1, x2, y2], axis=1)[geom_mask].astype(np.float32)
        return boxes, cand_scores[geom_mask].astype(np.float32)

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        if boxes.shape[0] == 0:
            return np.empty((0,), dtype=int)
        boxes_xywh = [
            [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes
        ]
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), self._obj_thresh, self._nms_thresh)
        if len(indices) == 0:
            return np.empty((0,), dtype=int)
        return np.array(indices, dtype=int).ravel()

    def _filter_head_texture(
        self, image: np.ndarray, boxes: np.ndarray, scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """High-confidence detections pass unconditionally. Below
        `high_conf_bypass`, a uniform (low-texture-STD) head/shoulder
        crop — a plain bag, coat, or poster rather than an actual head
        — gets discarded unless the score is high enough to trust
        anyway (`head_texture_score_thresh`).
        """
        if boxes.shape[0] == 0:
            return boxes, scores

        keep = []
        img_h, img_w = image.shape[:2]

        for i, (box, score) in enumerate(zip(boxes, scores)):
            if score >= self._high_conf_bypass:
                keep.append(i)
                continue

            x1, y1, x2, y2 = (int(round(v)) for v in box)
            bh = max(1, y2 - y1)
            head_y1 = max(0, y1)
            head_y2 = min(img_h, y1 + int(round(bh * 0.25)))
            head_x1 = max(0, x1)
            head_x2 = min(img_w, x2)

            if head_x2 > head_x1 and head_y2 > head_y1:
                head_crop = image[head_y1:head_y2, head_x1:head_x2]
                if head_crop.size > 0:
                    gray_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
                    std_dev = float(np.std(gray_head))
                    if std_dev < self._head_texture_std_thresh and score < self._head_texture_score_thresh:
                        continue

            keep.append(i)

        if not keep:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        idx = np.array(keep, dtype=int)
        return boxes[idx], scores[idx]

    def release(self) -> None:
        self._runtime.release()
