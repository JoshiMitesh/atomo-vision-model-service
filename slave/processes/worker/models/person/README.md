# Person model files

Drop the actual model weight/library files **directly in this folder**
(not inside `pt/`, `nb/`, `so/` — those hold worker/detector *code*
only):

```
processes/worker/models/person/
├── person.pt              ← .pt job weights
├── person.nb              ← .nb job model
├── person.so              ← .so job model, and/or the .nb companion library
│
├── pt/   person_pt_worker.py, person_pt_detector.py
├── nb/   person_nb_worker.py, person_nb_detector.py
└── so/   person_so_worker.py, person_so_detector.py
```

The file each worker loads is whatever `model_name` the job carries —
keep job messages / `config/jobs.yaml` entries using simple names that
match what's dropped here, e.g.:

```yaml
model_name: person.pt      # model_extension: .pt
model_name: person.nb      # model_extension: .nb
model_name: person.so      # model_extension: .so
```

For `.nb` jobs, the companion `.so` library filename is configured
separately in `config/models/person.yaml` under `nb.library_file`
(default `person.so`) and is also resolved against this same folder.
