"""Number-plate NPU pre/post-processing shared by the `.nb` and `.so`
detectors (they differ only in which runtime produces the raw output
tensors, like fire_nb_detector.py / fire_so_detector.py).

Ported from the board-verified alpr_npu.py (NPUDetector + decode_tensor
+ nms_boxes), which also settles what the compiled number_plate.nb
looks like:
  * input 640x640; frame is letterboxed (pad 114), BGR->RGB, /255,
    CHW, written into a reused (1, 3, 640, 640) float32 buffer, and
    NbRuntime's `reorder='2 1 0'` is still passed (that is exactly what
    the reference does -- Person's convention, not Fire's BGR one)
  * three outputs, finest first: 80x80, 40x40, 20x20 grids
  * 65 channels per cell, CLASSES FIRST: channel 0 = class logit,
    1..64 = 4 x 16 DFL logits (left, top, right, bottom)
  * per cell: sigmoid(class) >= candidate_gather_thresh, then
    DFL-certainty (mean peak softmax probability of the four groups)
    >= dfl_certainty_thresh, box = cell centre +/- expected DFL
    distance * stride, clipped to the 640 canvas
  * cv2 NMS with score threshold obj_thresh and IoU nms_thresh, then
    unletterbox and a plate-shape filter

The shape filter uses config/models/number_plate.yaml's `filters:`
block (pixels of the frame this worker actually receives), not the
reference's constants, which were tuned for its 464x832 downscaled
stream.

Only decode/filter happens here; tracking and OCR are in
npu_plate_enricher.py (CLAUDE.md SS8).
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from slave.logging.logger import get_logger
from slave.models.detection import Detection
from slave.runtimes.common import runtime_utils as ru

log = get_logger(__name__)

CLASS_NAME = "number_plate"
NUM_CLS = 1
REG_MAX = 16
LISTSIZE = NUM_CLS + 4 * REG_MAX  # 65


class PlateNpuDecoder:
    def __init__(self, model_config: dict, model_name: str) -> None:
        self._display_name = model_config.get("display_name") or model_name

        nb_cfg = model_config.get("nb", {})
        filters = model_config.get("filters", {})

        self._obj_thresh = float(nb_cfg.get("obj_thresh", 0.45))
        self._nms_thresh = float(nb_cfg.get("nms_thresh", 0.30))
        self._gather_thresh = float(nb_cfg.get("candidate_gather_thresh", 0.18))
        self._dfl_certainty_thresh = float(nb_cfg.get("dfl_certainty_thresh", 0.18))

        self._min_w = float(filters.get("min_plate_width_px", 45))
        self._min_h = float(filters.get("min_plate_height_px", 15))
        self._min_aspect = float(filters.get("min_plate_aspect", 2.0))
        self._max_aspect = float(filters.get("max_plate_aspect", 8.0))
        self._min_area_ratio = float(filters.get("min_plate_area_ratio", 0.00005))

        input_cfg = model_config.get("input", {})
        keyword = str(input_cfg.get("large_size_keyword", "1024"))
        self._input_size = int(
            input_cfg.get("large_size", 1024) if keyword in model_name else input_cfg.get("default_size", 640)
        )
        self._pad_value = int(input_cfg.get("pad_value", 114))
        self._lb_canvas = np.full((self._input_size, self._input_size, 3), self._pad_value, dtype=np.uint8)
        self._input_buf = np.zeros((1, 3, self._input_size, self._input_size), dtype=np.float32)
        self._shapes_logged = False

    def preprocess(self, frame: np.ndarray):
        letterboxed, info = ru.letterbox(frame, self._input_size, self._pad_value, canvas=self._lb_canvas)
        self._input_buf[0] = ru.to_input_tensor(letterboxed)  # BGR->RGB, CHW, /255
        return self._input_buf, info

    def decode(self, raw_outputs, info: ru.LetterboxInfo, frame_shape) -> List[Detection]:
        if raw_outputs is None or len(raw_outputs) < 3:
            return []

        # Finest grid first (80, 40, 20) -- the reference's verified order.
        outputs = sorted(raw_outputs[:3], key=lambda t: np.asarray(t).size, reverse=True)
        tensors = [ru.reshape_npu_output(raw, LISTSIZE)[0] for raw in outputs]  # each (LISTSIZE, H, W)

        if not self._shapes_logged:
            self._shapes_logged = True
            log.info(
                "number_plate NPU first inference: output grids=%s (expected %s)",
                [tuple(t.shape) for t in tensors],
                [(LISTSIZE, self._input_size // s, self._input_size // s) for s in (8, 16, 32)],
            )

        all_boxes, all_scores = [], []
        for tensor in tensors:
            boxes, scores = self._decode_grid(tensor, self._input_size / tensor.shape[-1])
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

        boxes = ru.unletterbox_boxes(boxes[keep], info)
        scores = scores[keep]

        img_h, img_w = frame_shape[:2]
        image_area = float(img_w * img_h)

        detections: List[Detection] = []
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = (float(v) for v in box)
            if not self._plate_shape_valid(x1, y1, x2, y2, image_area):
                continue
            detections.append(
                Detection(
                    class_id=0,
                    class_name=CLASS_NAME,
                    confidence=float(score),
                    bbox=(x1, y1, x2, y2),
                    model_name=self._display_name,
                )
            )
        return detections

    def _decode_grid(self, tensor: np.ndarray, stride: float) -> Tuple[np.ndarray, np.ndarray]:
        _, _, grid_w = tensor.shape
        flat = tensor.reshape(tensor.shape[0], -1)  # (LISTSIZE, N)

        scores = ru.sigmoid(np.clip(flat[0], -60.0, 60.0))  # channel 0 = class logit
        keep = scores >= self._gather_thresh
        if not np.any(keep):
            return self._empty()

        dfl = ru.softmax(flat[NUM_CLS:, keep].reshape(4, REG_MAX, -1), axis=1)  # (4, REG_MAX, K)

        certainty = dfl.max(axis=1).mean(axis=0)  # (K,)
        cert_keep = certainty >= self._dfl_certainty_thresh
        if not np.any(cert_keep):
            return self._empty()

        idx = np.nonzero(keep)[0][cert_keep]
        dfl = dfl[:, :, cert_keep]
        cand_scores = scores[keep][cert_keep]

        bins = np.arange(REG_MAX, dtype=np.float32)
        ltrb = np.tensordot(bins, dfl, axes=(0, 1)).T * stride  # (K, 4) pixels

        rows, cols = np.divmod(idx, grid_w)
        cx = (cols + 0.5) * stride
        cy = (rows + 0.5) * stride
        boxes = np.stack([cx - ltrb[:, 0], cy - ltrb[:, 1], cx + ltrb[:, 2], cy + ltrb[:, 3]], axis=1)
        boxes = np.clip(boxes, 0.0, float(self._input_size))

        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        if not np.any(valid):
            return self._empty()
        return boxes[valid].astype(np.float32), cand_scores[valid].astype(np.float32)

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
        indices = cv2.dnn.NMSBoxes(xywh, scores.tolist(), self._obj_thresh, self._nms_thresh)
        if indices is None or len(indices) == 0:
            return np.empty((0,), dtype=int)
        return np.array(indices, dtype=int).reshape(-1)

    def _plate_shape_valid(self, x1: float, y1: float, x2: float, y2: float, image_area: float) -> bool:
        w, h = x2 - x1, y2 - y1
        if w < self._min_w or h <= 0 or h < self._min_h:
            return False
        aspect = w / h
        if aspect < self._min_aspect or aspect > self._max_aspect:
            return False
        return (w * h) >= image_area * self._min_area_ratio

    @staticmethod
    def _empty() -> Tuple[np.ndarray, np.ndarray]:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
