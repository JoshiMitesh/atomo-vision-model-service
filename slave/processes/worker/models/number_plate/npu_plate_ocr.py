"""CPU-only PP-OCRv6 Medium recognition for NPU-detected plate crops."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from slave.logging.logger import get_logger

log = get_logger(__name__)


class NpuPlateOcr:
    """Recognize plate crops using the same adapter as the .pt path."""

    def __init__(self, model_name: str, engine: str, out_w: int, out_h: int) -> None:
        from slave.processes.worker.models.number_plate.ppocr_v6_tiny import (
            PpOcrV6Medium,
        )

        self._recognizer = PpOcrV6Medium(cpu_threads=2)
        log.info("NPU plate OCR ready (PP-OCRv6 Medium, ONNX Runtime, CPU)")

    def read(self, crop: np.ndarray) -> Tuple[str, float]:
        """Recognize the original detected crop without extra perspective warping."""
        try:
            if crop is None or crop.size == 0:
                return "", 0.0

            h, w = crop.shape[:2]
            text, score, valid, _ = self._recognizer.read_plate(
                crop, (0, 0, w, h)
            )

            if text:
                log.info(
                    "NPU plate OCR candidate: text=%r confidence=%.4f valid=%s",
                    text, score, valid,
                )

            # Preserve nonempty reads even if Indian-format validation fails.
            return text, score

        except Exception:
            log.exception("NPU plate OCR inference failed")
            return "", 0.0

    def release(self) -> None:
        if self._recognizer is not None:
            self._recognizer.release()
            self._recognizer = None
