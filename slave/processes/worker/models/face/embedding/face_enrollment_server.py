"""Internal HTTP endpoint for webcam face enrollment.

This server is deliberately separate from the live job workers. It reuses the
same FaceEnrollmentProcessor (YuNet + SFace) and therefore produces embeddings
compatible with live recognition.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from aiohttp import web

from slave.logging.logger import get_logger
from slave.processes.worker.models.face.embedding.face_enrollment import FaceEnrollmentProcessor

log = get_logger(__name__)


class FaceEnrollmentServer:
    def __init__(self, models_root: str) -> None:
        self._processor = FaceEnrollmentProcessor(models_root)
        self._runner: web.AppRunner | None = None
        self._thread: asyncio.AbstractEventLoop | None = None

    async def _handle(self, request: web.Request) -> web.Response:
        expected_key = os.getenv("FACE_ENROLLMENT_KEY", "").strip()
        if expected_key and request.headers.get("X-Enrollment-Key", "") != expected_key:
            return web.json_response({"ok": False, "reason": "unauthorized"}, status=401)

        try:
            body: dict[str, Any] = await request.json()
            frame = body.get("frame")
            if not isinstance(frame, str) or not frame:
                return web.json_response({"ok": False, "reason": "frame_required"}, status=400)
            result = await asyncio.to_thread(self._processor.process, frame)
            return web.json_response(result)
        except Exception:
            log.exception("face enrollment request failed")
            return web.json_response({"ok": False, "reason": "server_error"}, status=500)

    async def start(self) -> None:
        app = web.Application(client_max_size=2 * 1024 * 1024)
        app.router.add_get("/health", lambda request: web.json_response({"status": "ok"}))
        app.router.add_post("/internal/face-enrollment/frame", self._handle)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(
            self._runner,
            os.getenv("FACE_ENROLLMENT_BIND", "0.0.0.0"),
            int(os.getenv("FACE_ENROLLMENT_PORT", "8765")),
        )
        await site.start()
        log.info(
            "face enrollment service listening on %s:%s",
            os.getenv("FACE_ENROLLMENT_BIND", "0.0.0.0"),
            os.getenv("FACE_ENROLLMENT_PORT", "8765"),
        )

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None


def run_face_enrollment_server(models_root: str, stop_event) -> None:
    async def runner() -> None:
        server = FaceEnrollmentServer(models_root)
        await server.start()
        try:
            await asyncio.to_thread(stop_event.wait)
        finally:
            await server.stop()

    asyncio.run(runner())
