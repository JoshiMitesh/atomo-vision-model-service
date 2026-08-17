"""Detection — a single normalized detection box.

Every Person detector (.pt / .nb / .so) must translate its native model
output into this shape. It's the only contract shared across the three
implementations (see AGENTS.md §7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 in original frame pixels

    def to_dict(self) -> dict:
        return {
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "bbox": [float(v) for v in self.bbox],
        }
