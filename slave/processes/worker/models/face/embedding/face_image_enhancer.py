"""Lightweight same-size face enhancement for recognition and enrollment.

The enhancement is deliberately non-generative: it improves local contrast,
reduces sensor/compression noise, and applies mild unsharp masking without
upscaling or inventing facial detail.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np


def enhance_face(image: np.ndarray) -> np.ndarray:
    """Enhance a face while preserving its exact shape and dtype."""
    if image is None or image.size == 0:
        return image

    original_shape = image.shape
    original_dtype = image.dtype

    # SFace receives BGR images from OpenCV. Work in LAB so contrast is
    # improved mostly through luminance instead of changing skin colours.
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv.cvtColor(
        cv.merge((l_channel, a_channel, b_channel)),
        cv.COLOR_LAB2BGR,
    )

    # Preserve small facial details while suppressing JPEG/sensor noise.
    denoised = cv.GaussianBlur(enhanced, (0, 0), 0.55)

    # Mild unsharp mask. This is intentionally conservative because
    # aggressive sharpening can create false edges that hurt recognition.
    sharpened = cv.addWeighted(enhanced, 1.18, denoised, -0.18, 0)

    if sharpened.shape != original_shape:
        sharpened = cv.resize(
            sharpened,
            (original_shape[1], original_shape[0]),
            interpolation=cv.INTER_LINEAR,
        )

    if sharpened.dtype != original_dtype:
        sharpened = sharpened.astype(original_dtype)

    return sharpened
