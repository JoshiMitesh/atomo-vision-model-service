"""Per-job plate tracker based on the supplied ALPR reference.

A track keeps OCR reads for the same physical plate. The best value is
selected by whole-string vote: highest observation count first, then
average OCR confidence. This matches the reference behaviour and avoids
publishing every intermediate OCR guess as a separate event.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple

from slave.models.detection import Detection

Center = Tuple[float, float]


@dataclass
class PlateTrack:
    track_id: int
    center: Center
    missed: int = 0
    last_ocr_frame: int = -1_000_000
    current_text: str = ""
    current_score: float = 0.0
    confirmation_count: int = 0
    published: bool = False
    history: List[Tuple[str, float]] = field(default_factory=list)

    def record_ocr(self, text: str, score: float, max_history: int) -> None:
        if not text:
            return

        self.history.append((text, float(score)))
        if len(self.history) > max_history:
            self.history.pop(0)

        counts = Counter(text for text, _ in self.history)
        candidates = []

        for value, count in counts.items():
            scores = [score for text, score in self.history if text == value]
            avg_score = sum(scores) / len(scores)
            candidates.append((count, avg_score, value))

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        count, avg_score, best_text = candidates[0]

        self.current_text = best_text
        self.current_score = avg_score
        self.confirmation_count = count


def _center(bbox: Tuple[float, float, float, float]) -> Center:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _distance(a: Center, b: Center) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class PlateTracker:
    def __init__(self, max_match_distance: float, max_missed_frames: int) -> None:
        self._max_match_distance = max_match_distance
        self._max_missed_frames = max_missed_frames
        self._tracks: List[PlateTrack] = []
        self._next_id = 1

    def update(self, detections: List[Detection]) -> List[Tuple[Detection, PlateTrack]]:
        centers = [_center(d.bbox) for d in detections]
        possible = []

        for ti, track in enumerate(self._tracks):
            for di, center in enumerate(centers):
                possible.append((_distance(track.center, center), ti, di))

        possible.sort(key=lambda item: item[0])

        matched_tracks = set()
        matched_detections = set()
        result: List[Tuple[Detection, PlateTrack] | None] = [None] * len(detections)

        for distance, ti, di in possible:
            if distance > self._max_match_distance:
                continue
            if ti in matched_tracks or di in matched_detections:
                continue

            track = self._tracks[ti]
            track.center = centers[di]
            track.missed = 0
            matched_tracks.add(ti)
            matched_detections.add(di)
            result[di] = (detections[di], track)

        for ti, track in enumerate(self._tracks):
            if ti not in matched_tracks:
                track.missed += 1

        for di, detection in enumerate(detections):
            if di in matched_detections:
                continue

            track = PlateTrack(
                track_id=self._next_id,
                center=centers[di],
            )
            self._next_id += 1
            self._tracks.append(track)
            result[di] = (detection, track)

        self._tracks = [
            track for track in self._tracks
            if track.missed <= self._max_missed_frames
        ]

        return [item for item in result if item is not None]
