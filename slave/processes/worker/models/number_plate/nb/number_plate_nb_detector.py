"""Number-plate `.nb` detector -- Amlogic A311D NPU inference path.

Same shape as fire_nb_detector.py: this class only owns the NbRuntime
handle; preprocessing (BGR letterbox) and decode (sigmoid gate, DFL
certainty, DFL expectation, NMS, plate-shape filter) live in
npu_decoder.py so the `.so` detector can share them. See that module's
docstring for what was confirmed from the compiled number_plate.nb
(640 input, 3 grids x 65 channels) and what still needs first-run
confirmation on the board (channel order -- `nb.channel_layout`).

OCR and cross-frame plate tracking stay out of the detector
(plate_enricher.py, per-job, called by the worker) per CLAUDE.md SS8.
"""

from __future__ import annotations

from typing import List

import numpy as np

from slave.models.detection import Detection
from slave.processes.worker.models.number_plate.npu_decoder import PlateNpuDecoder
from slave.runtimes.npu.nb_runtime import NbRuntime


class NumberPlateNbDetector:
    def __init__(self, model_path: str, library_path: str, model_config: dict, model_name: str) -> None:
        self._decoder = PlateNpuDecoder(model_config, model_name)

        nb_cfg = model_config.get("nb", {})
        self._runtime = NbRuntime(
            model_path=model_path,
            library_path=library_path,
            level=int(nb_cfg.get("level", 0)),
            device_id=int(nb_cfg.get("device_id", 0)),
            board=str(nb_cfg.get("board", "Electron")),
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        input_tensor, info = self._decoder.preprocess(frame)
        raw_outputs = self._runtime.infer(input_tensor)
        return self._decoder.decode(raw_outputs, info, frame.shape)

    def release(self) -> None:
        self._runtime.release()
