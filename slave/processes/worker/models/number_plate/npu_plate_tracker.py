"""IoU plate tracker + evidence-weighted plate-text voting for the NPU
(`.nb`/`.so`) number_plate path.

Ported from the board-verified alpr_npu.py (bbox_iou, PlateTrack,
PlateTracker). Kept independent of plate_tracker.py, which the `.pt`
path uses (nearest-center matching + per-character consensus): the two
pipelines are tuned separately and must not affect each other.

One NpuPlateTracker per job, never shared (CLAUDE.md SS20).

How a plate number is chosen for a track: every OCR read is cleaned
(npu_plate_text.choose_best_plate_candidate) and given an evidence
weight = OCR confidence x detection confidence x (1.65 if valid Indian
format else 0.30) [x small bonuses for long plates]. Weights are summed
per distinct text; repeats and repeated valid reads boost a candidate,
valid-format candidates always outrank invalid ones. A plate is
"stable" only once it has >= min_valid_observations valid reads AND
total evidence >= min_stable_score.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from slave.models.detection import Detection
from slave.processes.worker.models.number_plate.npu_plate_text import (
    choose_best_plate_candidate,
    is_indian_plate,
    normalize_ocr,
)

Box = Tuple[float, float, float, float]


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class NpuPlateTrack:
    def __init__(
        self,
        track_id: int,
        box: Box,
        detection_conf: float,
        max_history: int,
        min_valid_observations: int,
        min_stable_score: float,
    ) -> None:
        self.track_id = track_id
        self.box = box
        self.detection_conf = detection_conf
        self.age = 1
        self.missed = 0
        self.history: List[dict] = []
        self.stable_text = ""
        self.published = False
        self.total_valid = 0
        self._max_history = max_history
        self._min_valid_observations = min_valid_observations
        self._min_stable_score = min_stable_score

    def update_box(self, box: Box, detection_conf: float) -> None:
        self.box = box
        self.detection_conf = detection_conf
        self.age += 1
        self.missed = 0

    def miss(self) -> None:
        self.missed += 1

    def add_ocr(self, raw_text: str, raw_conf: float) -> None:
        text, conf, valid = choose_best_plate_candidate(raw_text, raw_conf)
        if not text:
            return

        det_weight = max(0.35, min(1.0, float(self.detection_conf)))
        valid_bonus = 1.65 if valid else 0.30
        ocr_weight = max(0.10, min(1.0, float(conf)))
        weight = ocr_weight * det_weight * valid_bonus

        if valid:
            if len(text) >= 10:
                weight *= 1.10
            if len(text) >= 9:
                weight *= 1.05

        self.history.append(
            {
                "text": text,
                "raw": normalize_ocr(raw_text),
                "conf": float(conf),
                "valid": bool(valid),
                "weight": float(weight),
            }
        )
        if len(self.history) > self._max_history:
            self.history = self.history[-self._max_history:]

        if valid:
            self.total_valid += 1

    def get_best_result(self) -> Tuple[str, float, bool, float]:
        """Returns (text, avg_ocr_conf, stable, evidence_score)."""
        if not self.history:
            return "", 0.0, False, 0.0

        scores: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        valid_counts: Dict[str, int] = {}
        for item in self.history:
            text = item["text"]
            if not text:
                continue
            scores[text] = scores.get(text, 0.0) + item["weight"]
            counts[text] = counts.get(text, 0) + 1
            if item["valid"]:
                valid_counts[text] = valid_counts.get(text, 0) + 1

        if not scores:
            return "", 0.0, False, 0.0

        candidates = []
        for text, score in scores.items():
            count = counts.get(text, 0)
            valid_count = valid_counts.get(text, 0)
            is_valid = is_indian_plate(text)

            final_score = score
            if count >= 2:
                final_score *= 1.10
            if count >= 3:
                final_score *= 1.10
            if is_valid:
                final_score *= 1.45
            if valid_count >= 2:
                final_score *= 1.20
            if valid_count >= 3:
                final_score *= 1.15

            candidates.append((final_score, text, count, valid_count, is_valid))

        # Valid candidates first, then by number of valid reads, then score.
        candidates.sort(key=lambda x: (x[4], x[3], x[0]), reverse=True)
        final_score, text, _, valid_count, valid = candidates[0]

        stable = valid and valid_count >= self._min_valid_observations and final_score >= self._min_stable_score
        if stable:
            self.stable_text = text

        conf_values = [item["conf"] for item in self.history if item["text"] == text]
        avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0.0
        return text, avg_conf, stable, final_score


class NpuPlateTracker:
    def __init__(
        self,
        iou_threshold: float,
        max_missed: int,
        max_history: int,
        min_valid_observations: int,
        min_stable_score: float,
    ) -> None:
        self._iou_threshold = iou_threshold
        self._max_missed = max_missed
        self._track_kwargs = dict(
            max_history=max_history,
            min_valid_observations=min_valid_observations,
            min_stable_score=min_stable_score,
        )
        self._tracks: List[NpuPlateTrack] = []
        self._next_id = 1

    def _new_track(self, detection: Detection) -> NpuPlateTrack:
        track = NpuPlateTrack(self._next_id, detection.bbox, detection.confidence, **self._track_kwargs)
        self._next_id += 1
        return track

    def update(self, detections: List[Detection]) -> List[Tuple[Detection, NpuPlateTrack]]:
        """Match this frame's detections to tracks (greedy, highest IoU
        first) and return each detection paired with its track."""
        result: List[Optional[NpuPlateTrack]] = [None] * len(detections)

        pairs = []
        for ti, track in enumerate(self._tracks):
            for di, detection in enumerate(detections):
                iou = bbox_iou(track.box, detection.bbox)
                if iou >= self._iou_threshold:
                    pairs.append((iou, ti, di))
        pairs.sort(reverse=True)

        matched_tracks = set()
        matched_detections = set()
        for _, ti, di in pairs:
            if ti in matched_tracks or di in matched_detections:
                continue
            track = self._tracks[ti]
            track.update_box(detections[di].bbox, detections[di].confidence)
            result[di] = track
            matched_tracks.add(ti)
            matched_detections.add(di)

        for ti, track in enumerate(self._tracks):
            if ti not in matched_tracks:
                track.miss()

        for di, detection in enumerate(detections):
            if di not in matched_detections:
                track = self._new_track(detection)
                self._tracks.append(track)
                result[di] = track

        self._tracks = [t for t in self._tracks if t.missed <= self._max_missed]

        return [(d, t) for d, t in zip(detections, result) if t is not None]
