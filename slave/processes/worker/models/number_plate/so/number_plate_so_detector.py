"""Number-plate `.so` detector -- native shared-library inference path.

Same as fire_so_detector.py: identical preprocessing/decode to the
`.nb` path (shared via npu_decoder.py), but inference goes through
runtimes/npu/so_runtime.py (a directly-loaded native `.so` engine)
instead of a KSNN/ASNN `.nb` model.

Inherits so_runtime.py's current status: SoRuntime._init()/infer()
raise NotImplementedError until the board's real `.so` ABI is wired up
-- true for Person/Face/Safety/Fire's `.so` paths too, not a gap
introduced here. Note number_plate.so in this folder is the vsi_nn
companion library of number_plate.nb (exports vnn_CreateModel /
vnn_CreateNeuralNetwork), which is how the `.nb` path already uses it
via `nb.library_file`; a standalone `.so` job needs SoRuntime wired to
that ABI first.
"""

from __future__ import annotations

from typing import List

import numpy as np

from slave.models.detection import Detection
from slave.processes.worker.models.number_plate.npu_decoder import PlateNpuDecoder
from slave.runtimes.npu.so_runtime import SoRuntime


class NumberPlateSoDetector:
    def __init__(self, library_path: str, model_config: dict, model_name: str) -> None:
        self._decoder = PlateNpuDecoder(model_config, model_name)
        so_cfg = model_config.get("so", {})
        self._runtime = SoRuntime(library_path=library_path, device_id=int(so_cfg.get("device_id", 0)))

    def detect(self, frame: np.ndarray) -> List[Detection]:
        input_tensor, info = self._decoder.preprocess(frame)
        raw_outputs = self._runtime.infer(input_tensor)
        return self._decoder.decode(raw_outputs, info, frame.shape)

    def release(self) -> None:
        self._runtime.release()
