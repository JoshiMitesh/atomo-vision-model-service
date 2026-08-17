# Slave Node

Slave-side service for a distributed real-time video detection system.
Current scope: **Person Detection only** (`.pt`, `.nb`, `.so`), targeting
the Amlogic A311D NPU board. See `/CLAUDE.md`, `/AGENTS.md`, `/specs.md`
at the repository root for the full spec this implementation follows.

## Architecture

```
Master --(NATS job msg)--> subscriber -> job_store.py -> config/jobs.yaml
                                                              │
                                                     every 60s (job_watcher)
                                                              ▼
                                    job_manager -> model_manager -> model_resolver
                                                              │
                                                     model_registry -> worker_factory
                                                              │
                                                        Person Worker (.pt/.nb/.so)
                                                              │
                                     ┌────────────────────────┼────────────────────────┐
                                     ▼                        ▼                        ▼
                              NATS -> Master             MinIO frames            MinIO 30s video
```

The NATS subscriber only validates and persists job/stop messages — it
never starts inference directly. `job_watcher.py` polls `jobs.yaml`
every 60 seconds and reconciles running workers against it.

## Running

From the **repository root** (the parent of this `slave/` directory —
important, see below):

```bash
pip install -r slave/requirements.txt
python -m slave.main
```

Do not `cd slave && python main.py`. This package contains a
`slave/logging` module; running scripts from inside this directory
puts `slave/` itself on `sys.path` and shadows the standard library
`logging` module. Running as `python -m slave.main` from the parent
directory avoids that entirely.

## Configuration

- `config/slave.yaml` — process-level config: NATS, MinIO, logging, poll interval.
- `config/jobs.yaml` — runtime state (active jobs), written by the subscriber/stop handler.
- `config/models/person.yaml` — static Person model config (classes, thresholds, input size).
- `config/models/{face,fire,fight}.yaml` — placeholders only, not implemented.

## Model files

Drop weight/library files directly in
`processes/worker/models/person/` (e.g. `person.pt`, `person.nb`,
`person.so`) — see the README there for the full layout. `model_name`
in the job message must match the filename dropped there.

## Adding the Amlogic A311D NPU SDK

`.nb` and `.so` inference go through `runtimes/npu/nb_runtime.py` and
`runtimes/npu/so_runtime.py`. Both raise `NotImplementedError` at the
exact call sites where the board's real SDK/ABI needs to be wired in —
see the comments in each file. Do not reuse the reference
implementation's VIM3/KSNN or Electron/ASNN init values; they are
board-specific and wrong for A311D.

## Scope

Only Person Detection is implemented. Face/Fire/Fight folders exist as
placeholders only (`config/models/*.yaml`) and are rejected by
`model_resolver.py` until explicitly implemented.
