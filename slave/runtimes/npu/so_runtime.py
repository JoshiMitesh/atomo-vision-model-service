"""Amlogic A311D adapter for a standalone native `.so` inference library.

Distinct from nb_runtime.py: here `.so` IS the model/runtime (a
compiled inference engine loaded directly via ctypes), not the
companion library to a `.nb` file. Fill in the ctypes function
signatures to match the board's actual exported C ABI.
"""

from __future__ import annotations

import ctypes
from typing import List

import numpy as np

from slave.logging.logger import get_logger

log = get_logger(__name__)


class SoRuntime:
    def __init__(self, library_path: str, device_id: int = 0) -> None:
        self._library_path = library_path
        self._device_id = device_id
        self._lib: ctypes.CDLL | None = None
        self._init()

    def _init(self) -> None:
        self._lib = ctypes.CDLL(self._library_path)
        # NOTE: the exported symbol names/signatures are board/library
        # specific. Configure argtypes/restype for the real ABI, e.g.:
        #   self._lib.nn_init.argtypes = [ctypes.c_int]
        #   self._lib.nn_init.restype = ctypes.c_int
        #   if self._lib.nn_init(self._device_id) != 0:
        #       raise RuntimeError("nn_init failed")
        raise NotImplementedError(
            "SoRuntime._init: wire up the actual A311D .so ABI here "
            f"(library={self._library_path}, device_id={self._device_id})."
        )

    def infer(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        if self._lib is None:
            raise RuntimeError("SoRuntime used before successful init")
        # NOTE: replace with the real ctypes call into the .so, copying
        # input_tensor into the expected native buffer layout and
        # marshalling the output buffers back into numpy arrays.
        raise NotImplementedError("SoRuntime.infer: wire up the actual A311D .so inference call here")

    def release(self) -> None:
        if self._lib is not None:
            try:
                release_fn = getattr(self._lib, "nn_release", None)
                if release_fn:
                    release_fn()
            except Exception:
                log.exception("error releasing .so runtime handle")
            finally:
                self._lib = None
