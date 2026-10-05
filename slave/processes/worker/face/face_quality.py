"""Shared face detection quality gates."""

from __future__ import annotations

import math

from slave.models.detection import Detection


def passes_face_quality_gate(
    detection: Detection,
    *,
    min_confidence: float = 0.60,
    min_box_size: float = 60.0,
) -> bool:
    """Apply the moderate face-quality gate used before face tracking.

    Confidence/size checks apply to every face detector. YuNet landmark
    geometry is applied when the detector provides ``yunet_landmarks``.
    Detectors without YuNet landmarks are not rejected solely because those
    landmarks are unavailable.
    """
    if detection.confidence < min_confidence:
        return False

    x1, y1, x2, y2 = detection.bbox
    width = float(x2 - x1)
    height = float(y2 - y1)
    if width < min_box_size or height < min_box_size:
        return False

    landmarks = detection.extra.get("yunet_landmarks")
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

    margin_x = 0.05 * width
    margin_y = 0.05 * height
    for px, py in points:
        if px < x1 - margin_x or px > x2 + margin_x:
            return False
        if py < y1 - margin_y or py > y2 + margin_y:
            return False

    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    eye_gap = distance(right_eye, left_eye)
    mouth_gap = distance(right_mouth, left_mouth)
    if eye_gap < 0.08 * width or mouth_gap < 0.05 * width:
        return False

    eye_mid_x = (right_eye[0] + left_eye[0]) * 0.5
    eye_mid_y = (right_eye[1] + left_eye[1]) * 0.5
    mouth_mid_x = (right_mouth[0] + left_mouth[0]) * 0.5
    mouth_mid_y = (right_mouth[1] + left_mouth[1]) * 0.5

    if nose[1] <= eye_mid_y:
        return False
    if mouth_mid_y <= eye_mid_y + (0.12 * height):
        return False
    if mouth_mid_y <= nose[1] + (0.03 * height):
        return False

    eye_angle = math.degrees(
        math.atan2(
            right_eye[1] - left_eye[1],
            right_eye[0] - left_eye[0],
        )
    )
    if abs(eye_angle) > 20.0:
        return False

    nose_offset = abs(nose[0] - eye_mid_x) / max(eye_gap, 1.0)
    if nose_offset > 0.55:
        return False

    mouth_offset = abs(mouth_mid_x - eye_mid_x) / max(eye_gap, 1.0)
    if mouth_offset > 0.70:
        return False

    return True
