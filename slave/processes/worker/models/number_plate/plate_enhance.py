"""Night-plate crop cleaning: Lanczos upscale + light blur + black-hat.

Ported verbatim from anpr_night/plate_enhance.py -- proven on the
night-CCTV sample set (see anpr_night/results/, stream_out/). Used by
plate_ocr.py as one of the two crop variants ("clean") it runs OCR on
alongside the untouched ("raw") crop; per plate_enhance.py's own
finding, never apply this to a full frame before detection/cropping,
only to an already-cropped plate.
"""

from __future__ import annotations

import cv2
import numpy as np


def enhance_plate(crop_gray: np.ndarray, target_h: int = 64) -> np.ndarray:
    """Clean a plate crop for OCR. Input: grayscale crop cut from the RAW frame."""
    h = crop_gray.shape[0]
    s = max(1.0, target_h * 3 / h)                         # upscale small plates
    up = cv2.resize(crop_gray, None, fx=s, fy=s, interpolation=cv2.INTER_LANCZOS4)
    up = cv2.GaussianBlur(up, (0, 0), 0.8 * s / 4)         # remove JPEG blocks after upscale
    k = max(15, int(up.shape[0] * 0.13)) | 1               # wider than a stroke, narrower than the plate
    bh = cv2.morphologyEx(up, cv2.MORPH_BLACKHAT,          # keeps dark text, removes glare/gradient
                           cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return 255 - cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX)
