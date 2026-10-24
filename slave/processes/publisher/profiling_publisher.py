"""ProfilingPublisher writes telemetry payloads to local JSON log files.

The original implementation published the payload to a NATS subject. For the new
requirement we persist the data locally in a ``logs/profiling`` directory under
the repository root. Each payload is appended as a single JSON line (JSONL
format) to a file named ``<model_name>_YYYYMMDD.jsonl``. The directory is
created automatically if it does not exist.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

# Keeping compatibility with the original constructor signature.


class ProfilingPublisher:
    """Write profiling telemetry to local JSON log files.

    Parameters
    ----------
    publisher: Any
        Retained for compatibility; not used in the file‑logging version.
    profiling_subject: str
        Retained for compatibility; not used.
    """

    def __init__(self, publisher, profiling_subject: str) -> None:
        # Preserve the arguments for API compatibility but ignore them.
        self._publisher = publisher
        self._subject = profiling_subject
        # Resolve the repository root (two levels up from this file) and create
        # a ``logs/profiling`` directory.
        self._log_dir = Path(__file__).resolve().parents[2] / "logs" / "profiling"
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _log_path(self, model_name: str) -> Path:
        """Return the path of the log file for *model_name* for today.

        The file name format is ``<model_name>_YYYYMMDD.jsonl``.
        """
        date_str = datetime.utcnow().strftime("%Y%m%d")
        safe_name = model_name.replace(" ", "_") if model_name else "unknown"
        return self._log_dir / f"{safe_name}_{date_str}.jsonl"

    def publish(self, telemetry: dict) -> None:
        """Append the telemetry dictionary as a JSON line to the appropriate log.

        Errors are printed to ``stderr`` to avoid crashing the worker.
        """
        try:
            model_name = telemetry.get("model_name", "unknown")
            log_file = self._log_path(model_name)
            # Open in append‑binary mode to avoid encoding issues.
            with open(log_file, "ab") as f:
                line = json.dumps(telemetry, ensure_ascii=False).encode("utf-8") + b"\n"
                f.write(line)
        except Exception as exc:
            print(f"[ProfilingPublisher] failed to write telemetry: {exc}", file=os.sys.stderr)
