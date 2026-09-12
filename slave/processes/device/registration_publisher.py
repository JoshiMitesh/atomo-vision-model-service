"""One-time device registration/capability announcement.

Distinct from telemetry_publisher.py's periodic loop — this describes
what the device *is* (static identity + hardware spec + supported
model types), not its live metrics. Called once from device_agent.py
right after connect/subscribe, on devices.<device_name>.register.
"""

from __future__ import annotations

import json
import platform
import socket
import time
from typing import Any, Dict

from slave.config import DeviceConfig
from slave.logging.logger import get_logger

log = get_logger(__name__)


def build_registration_payload(device_config: DeviceConfig) -> Dict[str, Any]:
    return {
        "device_id": device_config.device_id or None,
        "device_name": device_config.device_name,
        "device_role": device_config.device_role,
        "hostname": socket.gethostname(),
        "ip": device_config.ip or None,
        "dns_name": device_config.dns_name or None,
        "mac_id": device_config.mac_id or None,
        "serial_no": device_config.serial_no or None,
        "approval_status": device_config.approval_status,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "hardware": {
            "soc": device_config.soc,
            "cpu": device_config.cpu_model,
            "npu_tops": device_config.npu_tops,
            "ram_gb": device_config.ram_gb,
            "storage": device_config.storage,
            "network": device_config.network,
        },
        "ai_capabilities": {
            "model_format": device_config.model_format,
            "supported_models": list(device_config.supported_models),
        },
        "status": "ONLINE",
        "ts": int(time.time() * 1000),
    }


async def publish_registration(nc, subject: str, device_config: DeviceConfig) -> None:
    payload = build_registration_payload(device_config)
    try:
        await nc.publish(subject, json.dumps(payload).encode())
        log.info("device registration published on %s", subject)
    except Exception:
        log.exception("failed to publish device registration on %s", subject)
