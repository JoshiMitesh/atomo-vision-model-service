"""EmbeddingDetector — wraps any Face detector (pt/nb/so) so every
`detect()` result also carries per-face SFace embeddings.

One wrapper covers both places a Face detector is used: the standalone
face_*_worker.py files and FaceController (via face_factory.py), so no
per-format embedding code exists. The wrapped detector is untouched —
it still only detects (CLAUDE.md §8).
"""

from __future__ import annotations

from typing import List

import numpy as np

from slave.models.detection import Detection
from slave.processes.worker.models.face.embedding.face_embedder import FaceEmbedder


class EmbeddingDetector:
    def __init__(self, detector, embedder: FaceEmbedder) -> None:
        self._detector = detector
        self._embedder = embedder

    def detect(self, frame: np.ndarray) -> List[Detection]:
        return self._embedder.attach(frame, self._detector.detect(frame))

    def release(self) -> None:
        try:
            self._detector.release()
        finally:
            self._embedder.release()


def with_embedding(
    detector,
    face_config: dict,
    models_dir: str,
    yunet_model_name: str | None = None,
):
    """Wrap `detector` with SFace embedding when face.yaml's
    `embedding.enabled` is true; otherwise return it unchanged.
    """
    embedding_cfg = face_config.get("embedding", {})
    if not bool(embedding_cfg.get("enabled", False)):
        return detector
    return EmbeddingDetector(
        detector,
        FaceEmbedder(embedding_cfg, models_dir, yunet_model_name=yunet_model_name),
    )
