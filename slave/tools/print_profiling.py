# file: /home/pal/Atomic_vision/atomo-vision-model-service/tools/print_profiling.py
# ---------------------------------------------------------------
# Small script that connects to the NATS server and prints every
# profiling message that arrives on the configured subject.
# ---------------------------------------------------------------
import asyncio
import os
import json
from nats.aio.client import Client as NATS
import sys, os, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

# Load the same config the slave uses (so we get the correct subject & URL)
from slave.config import load_slave_config

async def main():
    cfg = load_slave_config()
    nats_url = cfg.nats.url                # e.g. "nats://192.168.1.21:4222"
    profiling_subject = cfg.nats.profiling_subject   # e.g. "devices.neha-zbook.profiling"

    nc = NATS()
    await nc.connect(servers=[nats_url])

    async def message_handler(msg):
        data = msg.data.decode()
        try:
            payload = json.loads(data)
        except Exception:
            payload = data
        print(f"\n=== Profiling event received on {msg.subject} ===")
        print(json.dumps(payload, indent=2, sort_keys=True))

    # Subscribe – the subscription lives as long as the script runs
    await nc.subscribe(profiling_subject, cb=message_handler)

    print(f"Listening for profiling telemetry on [{profiling_subject}] …")
    # Keep the coroutine alive forever (or press Ctrl‑C to stop)
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
