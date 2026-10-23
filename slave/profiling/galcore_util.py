import pathlib

def _read_file(p: pathlib.Path) -> str | None:
    try:
        return p.read_text().strip()
    except Exception:
        return None

def _parse_percent(txt: str | None) -> float | None:
    if txt is None:
        return None
    txt = txt.rstrip('%')
    try:
        return float(txt)
    except ValueError:
        return None

# Primary debugfs location on Khadas VIM3
DEBUG_ROOT = pathlib.Path('/sys/kernel/debug/galcore')
_LOAD = DEBUG_ROOT / 'load'
_IDLE = DEBUG_ROOT / 'idle'

def read_npu_load() -> float | None:
    """Return NPU utilisation as a percentage (0‑100).
    Prefer the ``load`` file; if unavailable fall back to ``idle``
    (idle = 1‑load). Handles both percent and fractional representations.
    """
    raw = _read_file(_LOAD)
    pct = _parse_percent(raw)
    if pct is not None:
        return pct
    raw = _read_file(_IDLE)
    idle = _parse_percent(raw)
    if idle is None:
        return None
    # ``idle`` may be a fraction (0‑1) or a percent (>1)
    if idle > 1.0:
        idle = idle / 100.0
    return (1.0 - idle) * 100.0
