"""Per-job face tracker used to save each face crop once.

A detection matches a live track — and is treated as the same face,
NOT saved again — when its box overlaps the track's last box (IoU) or
either box's centre falls inside the other. A detection matching no
track starts a new one and is reported as new. A track unmatched for
`stale_seconds` is dropped, so the same face re-appearing later is
saved again.

Matching is by box overlap, not raw centre distance, so it tolerates
box jitter and size changes between processed frames. `stale_seconds`
uses wall-clock time rather than a frame/update count on purpose: this
pipeline hands the tracker whatever frame is newest when it gets to
it (frame_reader's queue drops stale frames under load — CLAUDE.md
§15), so consecutive update() calls can be over a second apart; a
count-based timeout would drop tracks far too eagerly under load.

One instance per worker — never shared between jobs (CLAUDE.md §20).
"""

from __future__ import annotations

import time
from typing import List

DEFAULT_IOU_THRESHOLD = 0.3
DEFAULT_STALE_SECONDS = 5.0


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _center_inside(box, other) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return other[0] <= cx <= other[2] and other[1] <= cy <= other[3]


def _matches(det_box, track_box, iou_threshold: float) -> bool:
    return (
        _iou(det_box, track_box) >= iou_threshold
        or _center_inside(det_box, track_box)
        or _center_inside(track_box, det_box)
    )


class FaceTracker:
    def __init__(
        self,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        stale_seconds: float = DEFAULT_STALE_SECONDS,
    ) -> None:
        self._iou_threshold = iou_threshold
        self._stale_seconds = stale_seconds
        self._tracks: List[dict] = []  # each: {"bbox": (...), "last_seen": float}

    def update(self, detections) -> list:
        """Feed one frame's detections (call every processed frame, even
        when empty). Returns only the detections that started a new
        track — these are the ones whose crop should be saved."""
        now = time.monotonic()
        matched = set()
        new_tracks = []
        new_dets = []

        for det in detections:
            best, best_score = None, 0.0
            for i, track in enumerate(self._tracks):
                if i in matched:
                    continue
                if not _matches(det.bbox, track["bbox"], self._iou_threshold):
                    continue
                score = _iou(det.bbox, track["bbox"])
                if best is None or score > best_score:
                    best, best_score = i, score
            if best is None:
                new_tracks.append({"bbox": tuple(det.bbox), "last_seen": now})
                new_dets.append(det)
            else:
                matched.add(best)
                self._tracks[best]["bbox"] = tuple(det.bbox)
                self._tracks[best]["last_seen"] = now

        self._tracks = [
            t for i, t in enumerate(self._tracks)
            if i in matched or now - t["last_seen"] < self._stale_seconds
        ]
        self._tracks.extend(new_tracks)
        return new_dets
