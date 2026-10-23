import pathlib

def _read_file(p: pathlib.Path) -> str | None:
    try:
        return p.read_text().strip()
    except PermissionError:
        # Fallback to sudo cat if the file is not readable by the current user
        import subprocess
        try:
            out = subprocess.check_output(['sudo', 'cat', str(p)], text=True)
            return out.strip()
        except Exception:
            return None
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
DEBUG_ROOTS = [
    pathlib.Path('/sys/kernel/debug/galcore'),
    pathlib.Path('/sys/kernel/debug/rknpu'),
]

def _find_file(name: str) -> pathlib.Path | None:
    """Return the first existing file matching *name* under any known DEBUG_ROOTS.
    ``name`` is a filename like ``load`` or ``idle``.
    """
    for root in DEBUG_ROOTS:
        candidate = root / name
        try:
            if candidate.exists():
                return candidate
        except PermissionError:
            # If we cannot stat the file, skip to next root – _read_file will try sudo.
            continue
    return None

def read_npu_load() -> float | None:
    """Return NPU utilisation as a percentage (0‑100).

    The function attempts to read a ``load`` or ``idle`` file from known debugfs
    locations. ``load`` reports utilisation directly; ``idle`` reports the idle
    fraction (idle = 1‑load). The implementation accepts both percent strings
    (e.g. ``"23%"``) and fractional values (e.g. ``"0.23"``). If none of the
    expected files are present, ``None`` is returned.
    """
    # Try a direct ``load`` file first
    load_path = _find_file('load')
    if load_path:
        raw = _read_file(load_path)
        pct = _parse_percent(raw)
        if pct is not None:
            return pct
    # Fallback to ``idle`` – convert idle fraction/percent to utilisation
    idle_path = _find_file('idle')
    if idle_path:
        raw = _read_file(idle_path)
        idle = _parse_percent(raw)
        if idle is not None:
            # ``idle`` may be a fraction (0‑1) or a percent (>1)
            if idle > 1.0:
                idle = idle / 100.0
            return (1.0 - idle) * 100.0
    # No usable debugfs entry – caller will see ``None``
    return None
