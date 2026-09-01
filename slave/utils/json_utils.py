"""JSON helpers that guarantee NumPy-free, JSON-native payloads.

Detection results are computed with NumPy internally but must never be
sent over NATS as NumPy objects — everything crossing that boundary
goes through ``to_json_safe``.
"""

from __future__ import annotations

import json
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a hard dependency at runtime
    np = None  # type: ignore[assignment]


def to_json_safe(value: Any) -> Any:
    """Recursively convert NumPy scalars/arrays (and other non-native
    types) into plain Python types suitable for ``json.dumps``.
    """
    if np is not None:
        if isinstance(value, np.ndarray):
            return [to_json_safe(v) for v in value.tolist()]
        if isinstance(value, np.generic):
            return value.item()
    if isinstance(value, dict):
        return {k: to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    return value


def dumps(value: Any) -> str:
    return json.dumps(to_json_safe(value))


def loads(raw: bytes | str) -> Any:
    return json.loads(raw)
