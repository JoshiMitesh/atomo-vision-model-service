"""Builds the RTSP stream URL from Master-provided connection info.

Kept isolated from rtsp_client.py so URL construction can change
(e.g. auth, alternate schemes) without touching capture/reconnect
logic (CLAUDE.md §14).
"""

from __future__ import annotations


def build_stream_url(master_ip: str, stream_port: int, stream_name: str) -> str:
    return f"rtsp://{master_ip}:{stream_port}/{stream_name}"
