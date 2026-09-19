"""Greedy nearest-center plate tracker, ported from the reference ALPR
script's PlateTrack/update_tracks (alpr_video.py) so OCR runs on the
same physical plate across frames instead of restarting every frame.

Model-backend-agnostic (works the same for pt/nb boxes) -- this is not
decode/NMS, so CLAUDE.md's detector-responsibility split does not
apply to it. It is a per-job, stateful worker concern: one PlateTracker
instance per job (via PlateEnricher), never shared across jobs
(CLAUDE.md SS20).

Unlike the reference, tracks aged out this frame are counted BEFORE
new tracks are appended, not after -- the reference's own loop order
would immediately increment a just-created track's `missed` to 1 in
the same frame it was created (its "not in matched_tracks" check does
not distinguish a genuinely-unmatched old track from a brand new one
appended earlier in the same call), which would wrongly skip that
track's very first OCR/publish opportunity. Fixed here since this is a
new module, not a port of NPU-specific behavior CLAUDE.md pins in
place.

Also unlike the reference/alpr_rtsp.py: PlateTrack resolves its
current_text via a score-weighted per-CHARACTER vote across every
accepted read seen so far, not a majority vote (or single best score)
on the whole string. Live testing against the real camera showed OCR
essentially never re-reads the identical full string twice even when
mostly correct -- consecutive reads of the same physical plate differ
by isolated single-character confusions (G/6, R/2, 8/0, etc.), usually
at a different position each time. Voting per character position lets
the positions every read agrees on lock in immediately, while an
isolated misread at one position in one frame gets outvoted by the
other frames that read that position correctly -- something neither
whole-string majority vote nor "keep the single highest-scoring read"
can do. It is NOT immune to a character the OCR model consistently
misreads across most/all frames of a given crop (a real limitation
seen in testing: if 2 of 3 reads share the same wrong digit at one
position, weighted vote reproduces that wrong digit too) -- fixing
that needs a sharper crop/better preprocessing, not a better voting
scheme; a single very-high-confidence outlier read can still win a
position against several lower-confidence wrong ones, but a
consistently-fooled OCR model cannot be voted around.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from slave.models.detection import Detection

Center = Tuple[float, float]


def _consensus(reads: List[Tuple[str, float]]) -> Tuple[str, float]:
    by_length: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
    for text, score in reads:
        by_length[len(text)].append((text, score))

    # Only same-length reads can be position-aligned -- trust whichever
    # length has the most total OCR confidence behind it, not just the
    # most observations.
    _, bucket = max(by_length.items(), key=lambda kv: sum(s for _, s in kv[1]))

    length = len(bucket[0][0])
    chars = []
    for pos in range(length):
        votes: Dict[str, float] = defaultdict(float)
        for text, score in bucket:
            votes[text[pos]] += score
        chars.append(max(votes.items(), key=lambda kv: kv[1])[0])

    avg_score = sum(s for _, s in bucket) / len(bucket)
    return "".join(chars), avg_score


@dataclass
class PlateTrack:
    track_id: int
    center: Center
    missed: int = 0
    last_ocr_frame: int = -1_000_000
    current_text: str = ""
    current_score: float = 0.0
    # Number of accepted (score + format valid) OCR reads seen so far
    # for this track -- gates whether PlateEnricher trusts current_text
    # enough to publish yet (config/models/number_plate.yaml's
    # `ocr.min_confirmations`).
    confirmation_count: int = 0
    _reads: List[Tuple[str, float]] = field(default_factory=list)

    def record_ocr(self, text: str, score: float, max_history: int) -> None:
        if not text:
            return
        self._reads.append((text, float(score)))
        if len(self._reads) > max_history:
            self._reads.pop(0)
        self.confirmation_count = len(self._reads)
        self.current_text, self.current_score = _consensus(self._reads)


def _center(bbox: Tuple[float, float, float, float]) -> Center:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _distance(a: Center, b: Center) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class PlateTracker:
    """One instance per job. update() takes this frame's raw plate
    Detection list and returns each paired with the PlateTrack it was
    matched or newly assigned to.
    """

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
        possible.sort(key=lambda p: p[0])

        matched_track_idx = set()
        matched_det_idx = set()
        result: List[Optional[Tuple[Detection, PlateTrack]]] = [None] * len(detections)

        for dist, ti, di in possible:
            if dist > self._max_match_distance:
                continue
            if ti in matched_track_idx or di in matched_det_idx:
                continue
            track = self._tracks[ti]
            track.center = centers[di]
            track.missed = 0
            matched_track_idx.add(ti)
            matched_det_idx.add(di)
            result[di] = (detections[di], track)

        for ti, track in enumerate(self._tracks):
            if ti not in matched_track_idx:
                track.missed += 1

        for di, detection in enumerate(detections):
            if di in matched_det_idx:
                continue
            track = PlateTrack(track_id=self._next_id, center=centers[di])
            self._next_id += 1
            self._tracks.append(track)
            result[di] = (detection, track)

        self._tracks = [t for t in self._tracks if t.missed <= self._max_missed_frames]

        return [r for r in result if r is not None]
