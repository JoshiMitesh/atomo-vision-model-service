# Face model files

Drop the actual model weight/library files **directly in this folder**
(not inside `pt/`, `nb/`, `so/` — those hold detector *code* only),
mirroring the Person model layout:

```
processes/worker/models/face/
├── face.pt                ← .pt face weights
├── face.nb                ← .nb face model
├── face.so                ← .so face model, and/or the .nb companion library
│
├── pt/   face_pt_detector.py
├── nb/   face_nb_detector.py
└── so/   face_so_detector.py
```

Unlike Person, Face is never named in `config/jobs.yaml` or a job
message — it is auto-triggered by the Person worker running on the
same job, whenever that worker sees a person (and for a configurable
grace window afterward). The Face model filename comes entirely from
`config/models/face.yaml` (`pt.model_file` / `nb.model_file` +
`nb.library_file` / `so.library_file`), always mirroring whichever
`model_extension` the job's Person model is using — e.g. a `.nb`
Person job runs Face via `.nb` too.

For `.nb` jobs, the companion `.so` library filename is configured in
`config/models/face.yaml` under `nb.library_file` (default `face.so`)
and is also resolved against this same folder.

## Face embeddings (SFace)

Every face detection also carries a 128-d SFace `embedding` (config:
`config/models/face.yaml` → `embedding:`), added by
`embedding/embedding_detector.py` around whichever detector
(pt/nb/so) is in use. Two extra ONNX files live flat in this folder:

```
face_recognition_sface_2021dec_int8.onnx   ← SFace embedding model
face_detection_yunet_2023mar_int8.onnx     ← YuNet, run on each face crop only for the 5 landmarks SFace alignment needs
```

Both run on CPU via OpenCV (`cv2.FaceRecognizerSF` / `cv2.FaceDetectorYN`,
OpenCV >= 4.5.4). A face with no landmarks, too small, or below
`min_det_conf` gets `embedding: null`. The Master does the matching.
