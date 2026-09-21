"""Draws detection boxes onto a copy of a frame for MinIO upload.

Kept separate from detectors (which must stay NumPy-in/Detection-out
only) and from frame_uploader.py (which stays a generic MinIO
primitive with no detection-specific knowledge). Workers call this
right before uploading — the raw frame handed to video recording is
left untouched.
"""

from __future__ import annotations

from typing import List, Sequence

import cv2
import numpy as np

from slave.models.detection import Detection

_BOX_COLOR = (0, 255, 0)
_BOX_THICKNESS = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.6
_FONT_THICKNESS = 2


def draw_detections(image: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    """Return a new image with each detection's box + label drawn on
    it. Does not modify ``image`` in place.
    """
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), _BOX_COLOR, _BOX_THICKNESS)

        label = f"{det.class_name} {det.confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, _FONT, _FONT_SCALE, _FONT_THICKNESS)
        label_y = max(y1, text_h + 8)
        cv2.rectangle(annotated, (x1, label_y - text_h - 8), (x1 + text_w + 4, label_y), _BOX_COLOR, -1)
        cv2.putText(annotated, label, (x1 + 2, label_y - 6), _FONT, _FONT_SCALE, (0, 0, 0), _FONT_THICKNESS)

    return annotated
