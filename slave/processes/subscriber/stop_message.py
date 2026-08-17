"""Parse/validate an incoming stop message.

Expected shape (see AGENTS.md §4):
    { "action": "stop", "job_id": "job_001" }
"""

from __future__ import annotations


class InvalidStopMessage(ValueError):
    pass


def parse_stop_message(raw: dict) -> str:
    """Return the job_id to stop, or raise InvalidStopMessage."""
    if not isinstance(raw, dict):
        raise InvalidStopMessage("stop message must be a JSON object")
    if raw.get("action") != "stop":
        raise InvalidStopMessage("stop message must set action='stop'")
    job_id = raw.get("job_id")
    if not job_id or not isinstance(job_id, str):
        raise InvalidStopMessage("stop message missing string job_id")
    return job_id
