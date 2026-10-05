"""Shared face detection quality gates."""

from __future__ import annotations

import math

from slave.models.detection import Detection


def passes_face_quality_gate(
    detection: Detection,
    *,
    min_confidence: float = 0.60,
    min_box_size: float = 50.0,
    min_brightness: float = 18.0,
    min_contrast: float = 8.0,
) -> bool:
    """Apply a moderate face-quality gate before tracking.

    The gate intentionally avoids strict pose checks so normal frontal and
    moderately angled faces remain detectable. YuNet landmark geometry is
    used only for basic face completeness/visibility checks. Recognition
    remains responsible for stricter embedding quality.
    """
    if detection.confidence < min_confidence:
        return False

    x1, y1, x2, y2 = detection.bbox
    width = float(x2 - x1)
    height = float(y2 - y1)
    if width < min_box_size or height < min_box_size:
        return False

    extra = detection.extra

    mean_brightness = extra.get("face_mean_brightness")
    if mean_brightness is not None:
        try:
            if float(mean_brightness) < min_brightness:
                return False
        except (TypeError, ValueError):
            return False

    contrast = extra.get("face_contrast")
    if contrast is not None:
        try:
            if float(contrast) < min_contrast:
                return False
        except (TypeError, ValueError):
            return False

    landmarks = extra.get("yunet_landmarks")
    if landmarks is None:
        return True

    try:
        values = [float(value) for value in landmarks]
    except (TypeError, ValueError):
        return False

    if len(values) != 10 or not all(math.isfinite(value) for value in values):
        return False

    right_eye = (values[0], values[1])
    left_eye = (values[2], values[3])
    nose = (values[4], values[5])
    right_mouth = (values[6], values[7])
    left_mouth = (values[8], values[9])
    points = (right_eye, left_eye, nose, right_mouth, left_mouth)

    margin_x = 0.12 * width
    margin_y = 0.12 * height
    for px, py in points:
        if px < x1 - margin_x or px > x2 + margin_x:
            return False
        if py < y1 - margin_y or py > y2 + margin_y:
            return False

    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    eye_gap = distance(right_eye, left_eye)
    mouth_gap = distance(right_mouth, left_mouth)
    if eye_gap < 0.05 * width or mouth_gap < 0.03 * width:
        return False

    eye_mid_y = (right_eye[1] + left_eye[1]) * 0.5
    mouth_mid_y = (right_mouth[1] + left_mouth[1]) * 0.5

    # Basic vertical face structure only. No strict yaw/roll/offset checks.
    if nose[1] < eye_mid_y - (0.05 * height):
        return False
    if mouth_mid_y < eye_mid_y + (0.05 * height):
        return False
    if mouth_mid_y < nose[1] - (0.02 * height):
        return False

    return True
