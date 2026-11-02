"""PP-OCRv6 Medium recognition-only adapter for already detected plate crops.

Requires paddleocr with ONNX Runtime inference support. Model weights are
downloaded by PaddleOCR on first use and cached for subsequent offline runs.
"""
from __future__ import annotations

import cv2
import numpy as np

from slave.processes.worker.models.number_plate.plate_format import normalize_plate


class PpOcrV6Medium:
    def __init__(self, cpu_threads: int = 2, **_kwargs) -> None:
        from paddleocr import TextRecognition
        # TextRecognition performs the model's own resize/normalization and
        # CTC decoding; don't manually decode ONNX outputs with a guessed charset.
        self._model = TextRecognition(
            model_name="PP-OCRv6_medium_rec",
            engine="onnxruntime",
            device="cpu",
        )

    def read_plate(self, frame: np.ndarray, bbox) -> tuple:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        x1, y1 = max(0, min(w, x1)), max(0, min(h, y1))
        x2, y2 = max(0, min(w, x2)), max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return "", 0.0, False, []
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "", 0.0, False, []
        # Upscale small/night crops and improve local contrast without
        # aggressive thresholding that can erase thin plate characters.
        ch, cw = crop.shape[:2]
        if ch < 48:
            target_w = max(cw, int(round(cw * (48.0 / max(1, ch)))))
            crop = cv2.resize(crop, (target_w, 48), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)

        # Retry the original crop only when enhancement is not confident.
        outputs = self._model.predict(input=enhanced, batch_size=1)
        for output in outputs:
            data = output.json
            if isinstance(data, dict) and "res" in data:
                data = data["res"]
            raw = str(data.get("rec_text", "") or "").strip().upper()
            score = float(data.get("rec_score", 0.0) or 0.0)
            if score < 0.95:
                fallback_outputs = self._model.predict(input=crop, batch_size=1)
                for fallback in fallback_outputs:
                    fallback_data = fallback.json
                    if isinstance(fallback_data, dict) and "res" in fallback_data:
                        fallback_data = fallback_data["res"]
                    fallback_raw = str(fallback_data.get("rec_text", "") or "").strip().upper()
                    fallback_score = float(fallback_data.get("rec_score", 0.0) or 0.0)
                    if fallback_score > score:
                        raw, score = fallback_raw, fallback_score
            plate, valid = normalize_plate(raw)
            return plate, score, valid, [(plate, valid, score, raw, "ppocr_v6_tiny", 0.0)]
        return "", 0.0, False, []

    def release(self) -> None:
        self._model = None
