import subprocess
from typing import Dict, Any

def read_gpu_stats() -> Dict[str, Any]:
    """Return GPU utilization info using nvidia-smi.
    Returns a dict with keys:
      - gpu_utilization (percent)
      - gpu_memory_total_mb
      - gpu_memory_used_mb
      - gpu_temperature_c
    If nvidia-smi is not available or fails, returns an empty dict.
    """
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.total,memory.used,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.check_output(cmd, text=True).strip()
        # Expected format: "0, 4096, 5, 42"
        parts = [p.strip() for p in result.split(",")]
        if len(parts) != 4:
            return {}
        util, mem_total, mem_used, temp = parts
        return {
            "gpu_utilization": float(util) if util else None,
            "gpu_memory_total_mb": float(mem_total) if mem_total else None,
            "gpu_memory_used_mb": float(mem_used) if mem_used else None,
            "gpu_temperature_c": float(temp) if temp else None,
        }
    except Exception:
        return {}
