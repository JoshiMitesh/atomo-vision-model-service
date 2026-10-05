"""Reference-compatible face frame/crop preprocessing.

The working atomo-ai-mitesh face pipeline processes camera frames at 960x540
and uses the same padded/CLAHE crop for recognition media.
"""
from __future__ import annotations

import cv2
import numpy as np

REFERENCE_WIDTH = 960
REFERENCE_HEIGHT = 540
FACE_MIN_CONFIDENCE = 0.60
FACE_MIN_SIZE = 50
FACE_CROP_PADDING = 0.60
FACE_CROP_MIN_SIZE = 150
FACE_JPEG_QUALITY = 90


def resize_reference_frame(frame_image: np.ndarray) -> np.ndarray:
    """Return the exact 960x540 working-project processing frame."""
    if frame_image is None or frame_image.size == 0:
        return frame_image
    if frame_image.shape[1] == REFERENCE_WIDTH and frame_image.shape[0] == REFERENCE_HEIGHT:
        return frame_image
    return cv2.resize(
        frame_image,
        (REFERENCE_WIDTH, REFERENCE_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )


def crop_face(frame_image: np.ndarray, bbox) -> np.ndarray | None:
    """Reference crop: 60% padding, min 150px, CLAHE."""
    if frame_image is None or frame_image.size == 0:
        return None

    height, width = frame_image.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None

    box_w = x2 - x1
    box_h = y2 - y1
    pad_w = int(box_w * FACE_CROP_PADDING)
    pad_h = int(box_h * FACE_CROP_PADDING)

    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(width, x2 + pad_w)
    y2 = min(height, y2 + pad_h)

    crop = frame_image[y1:y2, x1:x2].copy()
    crop_h, crop_w = crop.shape[:2]
    if crop_h < FACE_CROP_MIN_SIZE or crop_w < FACE_CROP_MIN_SIZE:
        scale_factor = max(
            FACE_CROP_MIN_SIZE / crop_w,
            FACE_CROP_MIN_SIZE / crop_h,
        )
        crop = cv2.resize(
            crop,
            (int(crop_w * scale_factor), int(crop_h * scale_factor)),
            interpolation=cv2.INTER_CUBIC,
        )

    try:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        crop = cv2.cvtColor(
            cv2.merge([clahe.apply(l), a, b]),
            cv2.COLOR_LAB2BGR,
        )
    except Exception:
        pass

    return crop
