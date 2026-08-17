"""Shared numeric/image helpers for the NPU (.nb/.so) Person pipelines.

Both person_nb_detector.py and person_so_detector.py need the same
letterbox preprocessing and DFL/NMS decode math (CLAUDE.md §11, §12).
Factoring it here avoids duplicating it in two detectors without
merging them into one universal detector class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxInfo:
    scale: float
    pad_x: float
    pad_y: float
    original_size: Tuple[int, int]  # (width, height)


def letterbox(image: np.ndarray, size: int, pad_value: int = 114) -> Tuple[np.ndarray, LetterboxInfo]:
    """Resize preserving aspect ratio onto a `size` x `size` canvas,
    padding with `pad_value`. Returns the padded image and the info
    needed to map boxes back to original coordinates.
    """
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_w, new_h = round(width * scale), round(height * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    pad_x = (size - new_w) / 2
    pad_y = (size - new_h) / 2
    x0, y0 = round(pad_x), round(pad_y)
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized

    return canvas, LetterboxInfo(scale=scale, pad_x=pad_x, pad_y=pad_y, original_size=(width, height))


def to_input_tensor(letterboxed_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 HWC -> RGB float32 CHW, normalized to [0, 1]."""
    rgb = cv2.cvtColor(letterboxed_bgr, cv2.COLOR_BGR2RGB)
    chw = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    return np.ascontiguousarray(chw)


def unletterbox_boxes(boxes_xyxy: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    """Map boxes from letterboxed pixel space back to the original frame."""
    if boxes_xyxy.size == 0:
        return boxes_xyxy
    boxes = boxes_xyxy.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - info.pad_x) / info.scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - info.pad_y) / info.scale
    width, height = info.original_size
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return boxes


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def decode_grid(
    tensor: np.ndarray,
    num_cls: int,
    reg_max: int,
    stride: int,
    obj_thresh: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode one output grid, shaped (1, num_cls + 4*reg_max, H, W),
    into (boxes_xyxy, scores) in letterboxed pixel space.

    Channel layout matches the reference Person model: `num_cls` class
    logits followed by 4 groups of `reg_max` DFL logits, ordered
    left/top/right/bottom (CLAUDE.md §9, §12).
    """
    tensor = np.squeeze(tensor, axis=0)  # (listsize, H, W)
    _, grid_h, grid_w = tensor.shape
    flat = tensor.reshape(tensor.shape[0], -1)  # (listsize, H*W)

    cls_logits = flat[:num_cls]  # (num_cls, N)
    scores = sigmoid(cls_logits).max(axis=0)  # (N,) — single-class model, but keep this general

    keep = scores > obj_thresh
    if not np.any(keep):
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    dfl = flat[num_cls:].reshape(4, reg_max, -1)[:, :, keep]  # (4, reg_max, K)
    dfl = softmax(dfl, axis=1)
    bins = np.arange(reg_max, dtype=np.float32)
    ltrb = np.tensordot(bins, dfl, axes=(0, 1)).T * stride  # (K, 4) in pixels

    rows, cols = np.divmod(np.nonzero(keep)[0], grid_w)
    cx = (cols + 0.5) * stride
    cy = (rows + 0.5) * stride

    x1 = cx - ltrb[:, 0]
    y1 = cy - ltrb[:, 1]
    x2 = cx + ltrb[:, 2]
    y2 = cy + ltrb[:, 3]
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    return boxes, scores[keep].astype(np.float32)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    """Standard greedy NMS. Returns indices of boxes to keep."""
    if boxes.shape[0] == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)

        order = order[1:][iou <= iou_thresh]

    return keep
