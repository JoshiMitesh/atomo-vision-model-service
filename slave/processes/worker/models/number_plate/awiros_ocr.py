"""Awiros PP-OCRv5 recognition for confirmed plate crops.

Optional dependency: install slave/requirements-paddle.txt on PT deployments.
Weights and dictionary are fetched from Hugging Face on first use, into
the existing number_plate model directory. NPU paths remain unaffected.
"""
from __future__ import annotations

import copy
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

_LOCK = threading.Lock()
_REPO = "Awiros/anpr-ocr"
_ARCH = {
    "model_type": "rec", "algorithm": "SVTR_HGNet", "Transform": None,
    "Backbone": {"name": "PPHGNetV2_B4", "text_rec": True},
    "Head": {
        "name": "MultiHead",
        "out_channels_list": {"CTCLabelDecode": 64, "NRTRLabelDecode": 67},
        "head_list": [
            {"CTCHead": {"Neck": {"name": "svtr", "dims": 120, "depth": 2,
                                  "hidden_dims": 120, "kernel_size": [1, 3],
                                  "use_guide": True}, "Head": {"fc_decay": 1e-05}}},
            {"NRTRHead": {"nrtr_dim": 384, "max_text_length": 25}},
        ],
    },
}


def _assets(directory: Path) -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download
    directory.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        paths = []
        for name in ("model.safetensors", "en_dict.txt"):
            target = directory / name
            if not target.is_file() or target.stat().st_size == 0:
                # huggingface_hub downloads into its cache atomically and copies
                # into local_dir; no separate partially written model is activated.
                hf_hub_download(repo_id=_REPO, filename=name, local_dir=str(directory))
            if not target.is_file() or target.stat().st_size == 0:
                raise RuntimeError(f"OCR asset unavailable: {target}")
            paths.append(target)
        return paths[0], paths[1]


def _ppocr(directory: Path):
    root = Path(os.environ.get("ATOMO_PADDLEOCR_DIR", directory / "PaddleOCR"))
    if not (root / "ppocr" / "__init__.py").is_file():
        if root.exists():
            raise RuntimeError(f"Invalid PaddleOCR source directory: {root}")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/PaddlePaddle/PaddleOCR.git", str(root)],
            check=True, timeout=180,
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


class AwirosPlateOcr:
    def __init__(self, model_dir: str, device: str = "cpu", cpu_threads: int = 2):
        import paddle
        from safetensors.numpy import load_file

        directory = Path(model_dir)
        weights, dictionary = _assets(directory)
        _ppocr(directory)
        from ppocr.modeling.architectures import build_model
        from ppocr.postprocess import build_post_process

        if device == "gpu" and paddle.is_compiled_with_cuda():
            paddle.set_device("gpu")
        else:
            paddle.set_device("cpu")
            # Avoid excessive OpenMP/Paddle thread contention with the live
            # detector and stream decoder. Keep this configurable per host.
            if hasattr(paddle, "set_num_threads"):
                paddle.set_num_threads(max(1, int(cpu_threads)))
        self._paddle = paddle
        self._model = build_model(copy.deepcopy(_ARCH))
        state = {key: paddle.to_tensor(value) for key, value in load_file(str(weights)).items()}
        self._model.set_state_dict(state)
        self._model.eval()
        self._decode = build_post_process({
            "name": "CTCLabelDecode",
            "character_dict_path": str(dictionary),
            "use_space_char": True,
        })

    def read_plate(self, frame: np.ndarray, bbox):
        x1, y1, x2, y2 = (int(round(x)) for x in bbox)
        h, w = frame.shape[:2]
        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if crop.size == 0:
            return "", 0.0, False, []
        height, width = crop.shape[:2]
        new_width = max(1, min(320, int(width * 48 / height)))
        resized = cv2.resize(crop, (new_width, 48))
        padded = np.zeros((48, 320, 3), dtype=np.uint8)
        padded[:, :new_width] = resized
        image = ((padded.astype(np.float32) / 255.0) - 0.5) / 0.5
        tensor = self._paddle.to_tensor(image.transpose(2, 0, 1)[None, ...])
        with self._paddle.no_grad():
            prediction = self._model(tensor)
        if isinstance(prediction, dict):
            prediction = prediction.get("ctc", next(iter(prediction.values())))
        elif isinstance(prediction, (tuple, list)):
            prediction = prediction[0]
        decoded = self._decode(prediction.numpy())
        raw, score = decoded[0] if decoded else ("", 0.0)
        plate = re.sub(r"[^A-Z0-9]", "", raw.upper())
        return plate, float(score), bool(plate), [(plate, bool(plate), float(score), raw, "raw", 0.0)]

    def release(self):
        self._model = None
