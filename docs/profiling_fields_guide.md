# 📊 Device Profiling Metrics — Field Guide

> This document explains every field published in the device profiling payload.  
> Intended for frontend developers displaying metrics and for end-users reading the dashboard.

---

## 🕒 Timestamps & Duration

| Field | Display Label | Unit | What It Means |
|-------|--------------|------|---------------|
| `timestamp` | Event Time | ISO datetime | When this profiling snapshot was captured on the device. |
| `duration_ms` | Frame Processing Time | ms | Total wall-clock time to process one frame end-to-end (from queue pick-up to publishing results). |
| `end_to_end_latency_ms` | End-to-End Latency | ms | Total time from camera capture to final result. `null` if not measured. |

**Healthy range:** `duration_ms` < 500 ms for real-time processing. Higher values mean the device is falling behind the camera frame rate.

---

## 🎬 Pipeline Stage Timings

These break down exactly *where* time is spent inside one frame's processing cycle.

| Field | Display Label | Unit | What It Means |
|-------|--------------|------|---------------|
| `camera_decode_ms` | Camera Decode | ms | Time spent decoding the raw RTSP video stream into a usable image frame. |
| `queue_wait_ms` | Queue Wait | ms | How long the frame waited in the buffer before the model picked it up. High values = model is slower than the camera. |
| `preprocess_ms` | Preprocessing | ms | Time to resize, normalize, and prepare the image before feeding it to the model. |
| `inference_ms` | Model Inference | ms | Time the AI model took to run detection on the image. This is the core computation step. |
| `postprocess_ms` | Postprocessing | ms | Time to decode model outputs, apply filters, run OCR (for number plates), tracking, etc. |

### 💡 How these add up
```
camera_decode_ms + queue_wait_ms + preprocess_ms + inference_ms + postprocess_ms ≈ duration_ms
```

### ⚠️ Warning Signs
- `queue_wait_ms` > 200 ms → Model is too slow; frames are piling up.
- `inference_ms` > 300 ms → Model running on CPU instead of GPU/NPU, or hardware is underpowered.
- `postprocess_ms` >> `inference_ms` → OCR or tracking is the bottleneck (common for number plate models).

---

## 🖥️ CPU Metrics

| Field | Display Label | Unit | What It Means |
|-------|--------------|------|---------------|
| `cpu_percent` | CPU Usage (Average) | % | Average CPU usage across all cores at the time of capture. |
| `cpu_per_core` | CPU Usage per Core | % | Individual load on each CPU core (Core 0, Core 1, …). Useful for detecting unbalanced load. |
| `system_load` | System Load (1m / 5m / 15m) | — | Linux load averages over 1, 5, and 15 minutes. A value equal to the number of CPU cores = 100% busy. |
| `context_switches` | Context Switches | count | Cumulative number of OS thread switches. Very high values indicate heavy multi-threading overhead. |
| `thread_cpu` | Per-Thread CPU Usage | % | CPU usage per individual thread (identified by thread ID). Useful for debugging which thread is causing spikes. |

### 💡 What good looks like
- `cpu_percent`: 40–70 % is healthy under load.
- `system_load[0]` (1-min average): Should not exceed the number of CPU cores (e.g., 8 for an 8-core CPU) consistently.

### ⚠️ Warning Signs
- `cpu_percent` consistently > 90 % → Device is CPU-saturated; consider GPU offloading or reducing camera resolution.
- Several cores at 100 % → CPU cores are maxed out.

---

## 🎮 GPU Metrics

| Field | Display Label | Unit | What It Means |
|-------|--------------|------|---------------|
| `gpu_utilization` | GPU Usage | % | How much of the GPU's compute capacity is being used. 0 % means the model is running on CPU. |
| `gpu_memory_used_mb` | GPU Memory Used | MB | GPU VRAM currently consumed. A loaded model typically uses 200–600 MB. |
| `gpu_memory_total_mb` | GPU Memory Total | MB | Total VRAM available on the GPU. |
| `gpu_temperature_c` | GPU Temperature | °C | Current GPU temperature. Important for thermal health monitoring. |

### 💡 What good looks like
- `gpu_utilization`: 10–60 % means the GPU is being used effectively.
- `gpu_memory_used_mb`: Should be well above 5 MB when a model is loaded (5 MB = no model on GPU).
- `gpu_temperature_c`: < 80 °C is safe; > 90 °C is a warning.

### ⚠️ Warning Signs
- `gpu_utilization`: 0 % AND `inference_ms` > 100 ms → Model is still running on CPU. Check `device` config.
- `gpu_memory_used_mb`: ~5 MB → Model not loaded on GPU.
- `gpu_temperature_c` > 85 °C → Thermal throttling risk.

---

## 🧠 NPU Metrics (Edge AI Accelerator)

| Field | Display Label | Unit | What It Means |
|-------|--------------|------|---------------|
| `npu_utilization` | NPU Usage | % | Usage of the dedicated Neural Processing Unit (e.g., Amlogic A311D NPU). `null` if no NPU or not measured. |

> The NPU is the dedicated AI chip on embedded boards. It handles model inference faster and more efficiently than the CPU for supported model formats (`.nb`, `.so`).

---

## 🧮 Memory Metrics

| Field | Display Label | Unit | What It Means |
|-------|--------------|------|---------------|
| `memory_rss_mb` | Process RAM (RSS) | MB | Total RAM used by the model service process (Resident Set Size — actual physical memory). |
| `memory_pss_mb` | Process RAM (PSS) | MB | Proportional RAM usage, accounting for shared memory. More accurate than RSS. |
| `memory_used_mb` | System RAM Used | MB | Total system RAM in use across all applications. |
| `memory_total_mb` | System RAM Total | MB | Total RAM available on the device. |
| `swap_used_mb` | Swap Used | MB | RAM that has been moved to disk (swap). Any significant value means the system is under memory pressure. |

### 💡 What good looks like
- `memory_pss_mb` / `memory_rss_mb`: Should be stable and not growing over time (memory leak indicator).
- `swap_used_mb`: Ideally 0 MB. Non-zero means the system is tight on RAM.

### ⚠️ Warning Signs
- `swap_used_mb` > 100 MB → System is under memory pressure; performance will degrade.
- `memory_rss_mb` growing over hours → Possible memory leak in the worker.

---

## 📋 Model Information

| Field | Display Label | What It Means |
|-------|--------------|---------------|
| `model_name` | Model | The AI model file running this job (e.g., `number_plate.pt`, `person.pt`). |
| `queue_lengths` | Queue Lengths | Number of frames waiting in each processing queue. Empty `{}` = no backlog (healthy). |

---

## 🚦 Quick Health Summary (for Dashboard Cards)

| Status | Condition | Color |
|--------|-----------|-------|
| 🟢 Healthy | `inference_ms` < 100, `cpu_percent` < 80, `gpu_utilization` > 5 (if GPU model) | Green |
| 🟡 Warning | `inference_ms` 100–300, `cpu_percent` 80–90, `swap_used_mb` > 0 | Yellow |
| 🔴 Critical | `inference_ms` > 500, `cpu_percent` > 95, `gpu_memory_used_mb` ~5 MB, `swap_used_mb` > 200 | Red |

---

## 📌 Glossary (Plain English for End Users)

| Term | Plain English Explanation |
|------|--------------------------|
| **Inference** | The AI model "thinking" — looking at a frame and finding objects in it. |
| **Preprocessing** | Preparing the image before the AI sees it (resize, colour adjustment, etc.). |
| **Postprocessing** | After the AI finds objects, extra work like reading number plates (OCR) or tracking objects across frames. |
| **Queue Wait** | How long a camera frame waited in line before being processed. Like a queue at a ticket counter. |
| **GPU** | Graphics Processing Unit — a chip that can run AI models much faster than a regular CPU. |
| **NPU** | Neural Processing Unit — a dedicated AI chip on embedded boards, even more efficient than a GPU for AI tasks. |
| **RSS Memory** | How much RAM the model service is using right now. |
| **Swap** | Slow emergency memory (disk) used when RAM runs out. Should always be 0. |
| **Context Switches** | How often the OS switches between tasks. High values = heavy multitasking overhead. |
| **System Load** | How busy the whole computer is overall. A load of 8 on an 8-core CPU = 100% busy. |
| **FP16 (half precision)** | A faster, more memory-efficient way to run AI models on a GPU. Cuts memory use in half with minimal accuracy loss. |

---

*Last updated: 2026-10-23 | Source: `slave/profiling/profiler.py` + `slave/profiling/gpu_util.py`*
