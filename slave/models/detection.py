"""Detection — a single normalized detection box.

Every detector (Person or Face; .pt / .nb / .so) must translate its
native model output into this shape. It's the only contract shared
across implementations (see AGENTS.md §7).

`model_name` identifies which model file produced this particular
detection — a single message's `detections` list can now mix results
from more than one model (e.g. Person + Face), so the model identity
had to move from the message's top level down onto each entry.
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
    model_name: str  # filename of the model that produced this detection

    def to_dict(self) -> dict:
        return {
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "event": f"{self.class_name}_detected",
            "confidence": float(self.confidence),
            "bbox": [float(v) for v in self.bbox],
            "model_name": self.model_name,
        }
