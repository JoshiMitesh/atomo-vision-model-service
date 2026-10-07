"""Padded, contrast-normalised face crop for MinIO upload.

Same crop the standalone face .pt/YuNet workers produce: 60% padding
around the detector box, minimum 150 px, CLAHE on the L channel.
"""

from __future__ import annotations

import cv2


def crop_face(frame_image, bbox):
    height, width = frame_image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width, int(x1)))
    y1 = max(0, min(height, int(y1)))
    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None

    pad_w = int((x2 - x1) * 0.60)
    pad_h = int((y2 - y1) * 0.60)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(width, x2 + pad_w)
    y2 = min(height, y2 + pad_h)

    crop = frame_image[y1:y2, x1:x2].copy()
    crop_h, crop_w = crop.shape[:2]
    min_size = 150
    if crop_h < min_size or crop_w < min_size:
        scale_factor = max(min_size / crop_w, min_size / crop_h)
        crop = cv2.resize(
            crop,
            (int(crop_w * scale_factor), int(crop_h * scale_factor)),
            interpolation=cv2.INTER_CUBIC,
        )

    try:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        crop = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
    except Exception:
        pass

    return crop
