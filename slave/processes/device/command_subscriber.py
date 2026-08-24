"""NATS request-reply glue for device commands.

Parses/validates the raw message and replies synchronously — actual
command logic lives in command_handler.py, kept separate so it's
testable without a NATS connection.
"""

from __future__ import annotations

import json

from slave.logging.logger import get_logger
from slave.processes.device.command_handler import execute_command

log = get_logger(__name__)


def build_command_handler(device_id: str):
    """Return an async NATS message callback bound to this device_id —
    used for logging context and passed through to execute_command()
    so status responses can identify which device answered.
    """

    async def _handler(msg) -> None:
        try:
            cmd = json.loads(msg.data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            await msg.respond(json.dumps({"status": "error", "detail": "bad payload"}).encode())
            return

        log.info("device_id=%s received command: %s", device_id, cmd)
        try:
            result = execute_command(cmd, device_id)
        except Exception:
            log.exception("command execution failed for device_id=%s", device_id)
            result = {"status": "error", "op": cmd.get("op"), "detail": "internal error"}

        await msg.respond(json.dumps(result).encode())

    return _handler
