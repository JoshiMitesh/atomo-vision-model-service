"""NATS request-reply glue for device commands.

Parses/validates the raw message and replies synchronously — actual
command logic lives in command_handler.py, kept separate so it's
testable without a NATS connection.
"""

from __future__ import annotations

import json
from typing import Optional

from slave.logging.logger import get_logger
from slave.processes.device.command_handler import execute_command

log = get_logger(__name__)


def build_command_handler(device_name: str, jobs_yaml_path: Optional[str] = None, ip: str = ""):
    """Return an async NATS message callback bound to this device_name —
    used for logging context; jobs_yaml_path/ip are passed through to
    execute_command() so get_status can report live running_cameras/
    loaded_models plus device identity.
    """

    async def _handler(msg) -> None:
        try:
            cmd = json.loads(msg.data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            await msg.respond(json.dumps({"status": "error", "detail": "bad payload"}).encode())
            return

        log.info("device_name=%s received command: %s", device_name, cmd)
        try:
            result = execute_command(cmd, jobs_yaml_path, device_name, ip)
        except Exception:
            log.exception("command execution failed for device_name=%s", device_name)
            result = {"status": "error", "op": cmd.get("op"), "detail": "internal error"}

        await msg.respond(json.dumps(result).encode())

    return _handler
