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


def letterbox(image: np.ndarray, size: int, pad_value: int = 114, canvas: np.ndarray | None = None) -> Tuple[np.ndarray, LetterboxInfo]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_w, new_h = round(width * scale), round(height * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if canvas is None:
        canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    else:
        canvas[:] = pad_value          # wipe and reuse instead of allocating new
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
    # Clip before exp() so an out-of-range raw logit (e.g. from a
    # decode/channel-layout mismatch on a not-yet-fully-verified .nb
    # model) saturates to a negligibly-small/near-1 value instead of
    # raising numpy's "overflow encountered in exp" RuntimeWarning.
    # sigmoid(-60) ~= 8.76e-27 -- not literally 0.0, but well below any
    # real obj/class threshold used anywhere in this codebase (all
    # >= 0.05), so this changes nothing for any normal-range logit.
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def reshape_npu_output(raw: np.ndarray, listsize: int) -> np.ndarray:
    """Normalize one raw NPU output tensor into (1, listsize, grid, grid)
    — the shape decode_grid() expects.

    The vendor SDK (KSNN/ASNN) does not guarantee outputs already come
    back in that shape — the reference NPU scripts on the board (e.g.
    `parse_npu_output()` in p2.py) reshape/transpose each raw tensor
    before decoding, and this is the same step. Handles tensors already
    shaped (1, listsize, H, W) or (listsize, H, W) or (H, W, listsize)
    unchanged; anything else (including flat) is reshaped assuming a
    square HxW grid, which holds for this Person model's three
    detection heads.
    """
    arr = np.asarray(raw, dtype=np.float32)

    if arr.ndim == 4 and arr.shape[1] == listsize:
        return arr
    if arr.ndim == 3:
        if arr.shape[0] == listsize:
            return arr[np.newaxis, ...]
        if arr.shape[-1] == listsize:
            return arr.transpose(2, 0, 1)[np.newaxis, ...]

    total = arr.size
    if total % listsize != 0:
        raise ValueError(f"NPU output size {total} not divisible by listsize {listsize}")
    grid_area = total // listsize
    grid = int(round(grid_area ** 0.5))
    if grid * grid != grid_area:
        raise ValueError(f"NPU output does not form a square grid: {grid_area} elements per channel")
    return arr.reshape(1, listsize, grid, grid)


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


def decode_grid_direct(
    tensor: np.ndarray,
    num_cls: int,
    stride: int,
    obj_thresh: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode one output grid, shaped (1, num_cls + 4, H, W), into
    (boxes_xyxy, scores) in letterboxed pixel space.

    Unlike decode_grid() above, this is for models that regress the
    four l/t/r/b box offsets *directly* (no DFL distribution over
    reg_max bins per side) — e.g. a YOLO export with `nms=False` but
    no anchor-free DFL head. Confirmed against a working reference
    decode (face_attrs_npu_rtsp.py's decode_scale()) for this exact
    face model: single class-score channel (sigmoid), then 4 raw
    offset channels, ordered left/top/right/bottom, each already in
    grid-cell units — multiplying by `stride` converts straight to
    letterboxed pixels, no softmax/bin-expectation step needed.
    """
    tensor = np.squeeze(tensor, axis=0)  # (listsize, H, W)
    _, grid_h, grid_w = tensor.shape

    cls_logits = tensor[:num_cls]  # (num_cls, H, W)
    scores = sigmoid(cls_logits).max(axis=0)  # (H, W)

    keep = scores > obj_thresh
    if not np.any(keep):
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    l = tensor[num_cls][keep]
    t = tensor[num_cls + 1][keep]
    r = tensor[num_cls + 2][keep]
    b = tensor[num_cls + 3][keep]

    rows, cols = np.nonzero(keep)
    cx = cols.astype(np.float32) + 0.5
    cy = rows.astype(np.float32) + 0.5

    x1 = (cx - l) * stride
    y1 = (cy - t) * stride
    x2 = (cx + r) * stride
    y2 = (cy + b) * stride
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


def decode_grid_multi(
    tensor: np.ndarray,
    num_cls: int,
    reg_max: int,
    stride: int,
    obj_thresh: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multi-class variant of decode_grid() — same DFL math, but also
    returns which class each box belongs to instead of assuming a
    single class.

    decode_grid() collapses the per-cell class scores to a single
    max-of-all-classes value (fine when num_cls==1, as for Person); a
    multi-class model (e.g. Safety's 9 classes) needs the actual
    argmax class id per detection too, or every box downstream would
    silently be reported under the wrong class_name.

    Returns (boxes_xyxy, scores, class_ids) in letterboxed pixel space.
    """
    tensor = np.squeeze(tensor, axis=0)  # (listsize, H, W)
    _, grid_h, grid_w = tensor.shape
    flat = tensor.reshape(tensor.shape[0], -1)  # (listsize, N)

    cls_logits = flat[:num_cls]  # (num_cls, N)
    cls_probs = sigmoid(cls_logits)  # (num_cls, N)
    class_ids = np.argmax(cls_probs, axis=0)  # (N,)
    scores = cls_probs[class_ids, np.arange(cls_probs.shape[1])]  # (N,) — score of the argmax class only

    keep = scores > obj_thresh
    if not np.any(keep):
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

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

    return boxes, scores[keep].astype(np.float32), class_ids[keep].astype(np.int64)


def nms_multiclass(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_thresh: float,
) -> List[int]:
    """Per-class NMS via the standard coordinate-offset trick: shift
    each box far enough (by class_id * (max_coord+1)) that boxes of
    different classes can never overlap, then reuse the existing
    class-agnostic nms(). Needed for Safety (multi-class) so an
    overlapping "person" box and "no_hardhat" box on the same head
    never wrongly suppress each other — plain nms() (as used by
    Person/Face, both single-class) would silently drop one of them.
    """
    if boxes.shape[0] == 0:
        return []
    max_coord = float(boxes.max()) if boxes.size else 0.0
    offsets = class_ids.astype(np.float32) * (max_coord + 1.0)
    boxes_offset = boxes + offsets[:, None]
    return nms(boxes_offset, scores, iou_thresh)
