"""Amlogic A311D NPU adapter for `.nb` model + `.so` library inference.

The reference implementation (`rtsp_30fps_realtime(1).py`) initializes
KSNN/ASNN with values tuned for VIM3/Electron boards. Those values must
NOT be reused for A311D (CLAUDE.md §10) — this module is the single
place where the actual A311D-installed SDK is loaded and called, so
swapping SDKs never touches detector code.

Fill in `_load_sdk()` / `_init_impl()` / `_infer_impl()` with the
board's real API once the A311D SDK is available in this environment.
"""

from __future__ import annotations

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
    for module_name in ("ksnn.api", "asnn.api"):
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

    def __init__(self, model_path: str, library_path: str | None, level: int = 0, device_id: int = 0) -> None:
        self._model_path = model_path
        self._library_path = library_path
        self._level = level
        self._device_id = device_id
        self._handle: Any = None
        self._init()

    def _init(self) -> None:
        sdk = _load_sdk()
        # NOTE: the exact constructor/init call is board-SDK specific.
        # Replace this with the real A311D API, e.g. something shaped like:
        #   self._handle = sdk.KSNN_API()  # or vendor equivalent
        #   self._handle.nn_init(library=self._library_path, model=self._model_path, level=self._level)
        raise NotImplementedError(
            "NbRuntime._init: wire up the actual A311D NPU SDK call here "
            f"(model={self._model_path}, library={self._library_path}, "
            f"level={self._level}, device_id={self._device_id})."
        )

    def infer(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """Run inference and return the raw output tensors (unsorted,
        undecoded) — decoding is the detector's responsibility.
        """
        if self._handle is None:
            raise NpuSdkUnavailableError("NbRuntime used before successful init")
        # NOTE: replace with the real inference call, e.g.:
        #   return self._handle.nn_inference([input_tensor], platform="ONNX", reorder="2 1 0", ...)
        raise NotImplementedError("NbRuntime.infer: wire up the actual A311D NPU inference call here")

    def release(self) -> None:
        if self._handle is not None:
            try:
                release_fn = getattr(self._handle, "release", None) or getattr(self._handle, "nn_release", None)
                if release_fn:
                    release_fn()
            except Exception:
                log.exception("error releasing NPU handle")
            finally:
                self._handle = None
