"""Shared face detection quality gates."""

from __future__ import annotations

import math

from slave.models.detection import Detection


def passes_face_quality_gate(
    detection: Detection,
    *,
    min_confidence: float = 0.70,
    min_box_size: float = 70.0,
    min_brightness: float = 25.0,
    min_contrast: float = 12.0,
) -> bool:
    """Accept clean, usable recognition-quality face crops with moderate pose tolerance.

    Detection is deliberately stricter than a generic face detector here.
    Side profiles, strong yaw/roll, downward/upward-looking poses, weak
    facial geometry, heavy occlusion and low-detail/motion-blurred crops must
    never enter tracking, clustering or the recognition gallery.
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
    # A recognition crop must contain both eyes with useful separation.
    # Small eye gaps are a strong signal of a side/three-quarter profile.
    if eye_gap < 0.16 * width or mouth_gap < 0.07 * width:
        return False

    eye_mid_x = (right_eye[0] + left_eye[0]) * 0.5
    eye_mid_y = (right_eye[1] + left_eye[1]) * 0.5
    nose_x, nose_y = nose
    mouth_mid_x = (right_mouth[0] + left_mouth[0]) * 0.5
    mouth_mid_y = (right_mouth[1] + left_mouth[1]) * 0.5
    face_mid_x = (x1 + x2) * 0.5

    # Reject strong yaw: landmarks must remain close to the face center.
    if abs(eye_mid_x - face_mid_x) > 0.16 * width:
        return False
    if abs(nose_x - eye_mid_x) > 0.13 * width:
        return False
    if abs(mouth_mid_x - eye_mid_x) > 0.14 * width:
        return False

    # Reject rolled/tilted faces.
    eye_dx = abs(left_eye[0] - right_eye[0])
    eye_dy = abs(left_eye[1] - right_eye[1])
    if eye_dx < 1e-6 or eye_dy / eye_dx > 0.30:
        return False

    # Strict vertical geometry rejects faces looking substantially up/down.
    eye_to_nose = (nose_y - eye_mid_y) / max(height, 1.0)
    nose_to_mouth = (mouth_mid_y - nose_y) / max(height, 1.0)
    eye_to_mouth = (mouth_mid_y - eye_mid_y) / max(height, 1.0)
    if not 0.14 <= eye_to_nose <= 0.44:
        return False
    if not 0.07 <= nose_to_mouth <= 0.28:
        return False
    if not 0.27 <= eye_to_mouth <= 0.62:
        return False

    # Require a complete, reasonably shaped face crop.
    face_aspect = width / max(height, 1.0)
    if not 0.60 <= face_aspect <= 1.30:
        return False

    return True
