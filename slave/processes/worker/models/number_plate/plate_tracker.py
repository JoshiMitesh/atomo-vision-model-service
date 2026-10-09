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

    # Capture-time state. PT reads the newest available frame, so processed
    # frame numbers are not consecutive source frames.
    last_seen_frame: int = -1
    last_seen_timestamp: float = 0.0
    velocity: Center = (0.0, 0.0)

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
    def __init__(
        self,
        max_match_distance: float,
        max_missed_frames: int,
        max_track_gap_seconds: float = 1.5,
    ) -> None:
        self._max_match_distance = max_match_distance
        self._max_missed_frames = max_missed_frames
        self._max_track_gap_seconds = max(0.1, float(max_track_gap_seconds))
        self._tracks: List[PlateTrack] = []
        self._next_id = 1

    @staticmethod
    def _age_seconds(track: PlateTrack, timestamp: float) -> float:
        if track.last_seen_timestamp <= 0.0 or timestamp <= 0.0:
            return 0.0
        return max(0.0, timestamp - track.last_seen_timestamp)

    @classmethod
    def _predicted_center(cls, track: PlateTrack, timestamp: float) -> Center:
        dt = cls._age_seconds(track, timestamp)
        return (
            track.center[0] + track.velocity[0] * dt,
            track.center[1] + track.velocity[1] * dt,
        )

    def update(
        self,
        detections: List[Detection],
        frame_number: int,
        timestamp: float,
    ) -> List[Tuple[Detection, PlateTrack]]:
        centers = [_center(d.bbox) for d in detections]
        possible = []

        for ti, track in enumerate(self._tracks):
            age = self._age_seconds(track, timestamp)
            if age > self._max_track_gap_seconds:
                continue

            predicted = self._predicted_center(track, timestamp)
            for di, center in enumerate(centers):
                distance = _distance(predicted, center)
                speed = (track.velocity[0] ** 2 + track.velocity[1] ** 2) ** 0.5
                motion_allowance = min(speed * age * 1.25, 500.0)
                possible.append((max(0.0, distance - motion_allowance), ti, di))

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
            old_center = track.center
            dt = self._age_seconds(track, timestamp)
            predicted = self._predicted_center(track, timestamp)
            actual_distance = _distance(predicted, centers[di])
            speed = (track.velocity[0] ** 2 + track.velocity[1] ** 2) ** 0.5
            allowed_distance = self._max_match_distance + min(speed * dt * 1.25, 500.0)
            if actual_distance > allowed_distance:
                continue

            if dt > 0.01:
                measured_velocity = (
                    (centers[di][0] - old_center[0]) / dt,
                    (centers[di][1] - old_center[1]) / dt,
                )
                alpha = 0.5
                track.velocity = (
                    alpha * measured_velocity[0] + (1.0 - alpha) * track.velocity[0],
                    alpha * measured_velocity[1] + (1.0 - alpha) * track.velocity[1],
                )

            track.center = centers[di]
            track.last_seen_frame = frame_number
            track.last_seen_timestamp = timestamp
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
                last_seen_frame=frame_number,
                last_seen_timestamp=timestamp,
            )
            self._next_id += 1
            self._tracks.append(track)
            result[di] = (detection, track)

        self._tracks = [
            track for track in self._tracks
            if (
                (timestamp <= 0.0 and track.missed <= self._max_missed_frames)
                or (timestamp > 0.0 and self._age_seconds(track, timestamp) <= self._max_track_gap_seconds)
            )
        ]

        return [item for item in result if item is not None]
