"""Frame — one image pulled off a job's RTSP stream."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    job_id: str
    frame_id: int
    image: np.ndarray
    timestamp: float
