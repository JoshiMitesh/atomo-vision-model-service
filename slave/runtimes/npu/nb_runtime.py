"""Amlogic A311D NPU adapter for `.nb` model + `.so` library inference.

The reference implementation (`rtsp_30fps_realtime(1).py`) initializes
KSNN/ASNN with values tuned for VIM3/Electron boards. Those values must
NOT be blindly reused for A311D (CLAUDE.md §10) — this module is the
single place where the actual A311D-installed SDK is loaded and
called, so swapping SDKs never touches detector code.

For *this* physically-confirmed board (hostname `electron`, A311D
running the ASNN SDK — see `~/detection.py`, `~/p2.py`,
`~/face_attrs_npu_rtsp.py` on the board, all of which use
`from asnn.api import asnn` and call `asnn("Electron")`), ASNN is the
real, working API — not a placeholder value copied from the reference.
KSNN is tried first only because the code is written to work on either
SDK; it isn't installed on this board and will simply fall through to
ASNN. If this runtime is ever deployed to a KSNN-only board, `board`
below should be overridden accordingly (it's only meaningful for ASNN).
"""

from __future__ import annotations

import importlib
from typing import Any, List

import numpy as np

from slave.logging.logger import get_logger

log = get_logger(__name__)


class NpuSdkUnavailableError(RuntimeError):
    """Raised when no A311D NPU SDK (KSNN/ASNN or equivalent) is
    importable in this environment. This is intentionally NOT silently
    swallowed — faking NPU output would be worse than failing loudly.
    """


def _load_sdk() -> Any:
    """Locate the A311D-installed NPU SDK. Try the vendor's actual
    package name(s) for this board; extend this list with whatever the
    A311D BSP/SDK documentation specifies.
    """
    for module_name in ("asnn.api", "ksnn.api"):
        try:
            module = __import__(module_name, fromlist=["*"])
            log.info("loaded NPU SDK module '%s'", module_name)
            return module
        except ImportError:
            continue
    raise NpuSdkUnavailableError(
        "No A311D NPU SDK found on this system. Install the board's NPU "
        "SDK (KSNN/ASNN or the vendor-provided equivalent) and, if its "
        "module name differs, add it to nb_runtime._load_sdk()."
    )


class NbRuntime:
    """Loads a .nb model (optionally paired with a .so library) and
    runs inference on the A311D NPU.
    """

    def __init__(
        self,
        model_path: str,
        library_path: str | None,
        level: int = 0,
        device_id: int = 0,
        board: str = "Electron",
    ) -> None:
        self._model_path = model_path
        self._library_path = library_path
        self._level = level
        self._device_id = device_id
        self._board = board
        self._handle: Any = None
        self._types: Any = None
        self._init()

    def _init(self) -> None:
        sdk = _load_sdk()
        self._types = importlib.import_module(sdk.__name__.rsplit(".", 1)[0] + ".types")

        if sdk.__name__.startswith("ksnn"):
            self._handle = sdk.KSNN()
        else:
            # ASNN's constructor takes the board identifier (see
            # ~/p2.py on the board: `asnn("Electron")`), unlike KSNN.
            self._handle = sdk.asnn(self._board)

        result = self._handle.nn_init(library=self._library_path, model=self._model_path, level=self._level)
        # The on-board reference scripts treat a result without 'SUCCESS' as a failed init.
        if result is not None and "SUCCESS" not in str(result):
            log.warning("NPU nn_init returned an unexpected result: %r (model=%s)", result, self._model_path)
        log.info(
            "NPU handle initialized (sdk=%s, board=%s, model=%s, library=%s, level=%s)",
            sdk.__name__, self._board, self._model_path, self._library_path, self._level,
        )

    def infer(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """Run inference and return the raw output tensors (unsorted,
        undecoded) — decoding is the detector's responsibility.
        """
        if self._handle is None:
            raise NpuSdkUnavailableError("NbRuntime used before successful init")
        data = self._handle.nn_inference(
            [input_tensor],
            platform="ONNX",
            reorder="2 1 0",
            output_tensor=3,
            output_format=self._types.output_format.OUT_FORMAT_FLOAT32,
        )
        return list(data)

    def release(self) -> None:
        if self._handle is not None:
            try:
                release_fn = (
                    getattr(self._handle, "release", None)
                    or getattr(self._handle, "nn_release", None)
                    or getattr(self._handle, "nn_deinit", None)  # what the on-board reference scripts call
                )
                if release_fn:
                    release_fn()
            except Exception:
                log.exception("error releasing NPU handle")
            finally:
                self._handle = None
