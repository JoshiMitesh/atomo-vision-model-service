"""build_face_controller — resolves the Face model for a Person job and
returns a ready-to-use FaceController.

Deliberately standalone from model_resolver.py/model_registry.py (the
job-driven model dispatch layer): Face is never a jobs.yaml entry, it
is only ever triggered in-process by the Person worker that owns the
job. This module is the one place that knows how to turn a Person
job's model_extension into a Face detector class + model path,
mirroring (not sharing) the pattern model_resolver.py uses for Person:

    person + .pt -> processes/worker/models/person/pt/
    face   + .pt -> processes/worker/models/face/pt/   (same mirrored extension)

config/models/face.yaml is loaded once per process and cached — every
job's FaceController reads the same static config, same as
config/models/person.yaml today.
"""

from __future__ import annotations

import os
import threading

import yaml

from slave.processes.worker.face.face_controller import FaceController

_face_config_cache: dict = {}
_cache_lock = threading.Lock()


def _load_face_config(config_root: str) -> dict:
    path = os.path.join(config_root, "models", "face.yaml")
    with _cache_lock:
        if path not in _face_config_cache:
            with open(path, "r", encoding="utf-8") as fh:
                _face_config_cache[path] = yaml.safe_load(fh) or {}
        return _face_config_cache[path]


def _build_detector(model_extension: str, models_root: str, face_cfg: dict):
    # Face binaries live flat in processes/worker/models/face/ (same
    # convention as person's models/person/README.md), sibling to the
    # pt/nb/so *code* subfolders.
    face_models_dir = os.path.join(models_root, "face")

    if model_extension == ".pt":
        from slave.processes.worker.models.face.pt.face_pt_detector import FacePtDetector

        pt_cfg = face_cfg.get("pt", {})
        model_name = pt_cfg.get("model_file", "face.pt")
        model_path = os.path.join(face_models_dir, model_name)
        return FacePtDetector(model_path, face_cfg, model_name)

    if model_extension == ".nb":
        from slave.processes.worker.models.face.nb.face_nb_detector import FaceNbDetector

        nb_cfg = face_cfg.get("nb", {})
        model_name = nb_cfg.get("model_file", "face.nb")
        model_path = os.path.join(face_models_dir, model_name)
        library_file = nb_cfg.get("library_file", "")
        library_path = os.path.join(face_models_dir, library_file) if library_file else None
        return FaceNbDetector(model_path, library_path, face_cfg, model_name)

    if model_extension == ".so":
        from slave.processes.worker.models.face.so.face_so_detector import FaceSoDetector

        so_cfg = face_cfg.get("so", {})
        library_file = so_cfg.get("library_file", "face.so")
        library_path = os.path.join(face_models_dir, library_file)
        return FaceSoDetector(library_path, face_cfg, library_file)

    raise ValueError(f"face detection: unsupported model_extension '{model_extension}'")


def build_face_controller(
    job_id: str,
    model_extension: str,
    models_root: str,
    config_root: str,
) -> FaceController:
    face_cfg = _load_face_config(config_root)
    trigger_cfg = face_cfg.get("trigger", {})

    def detector_factory():
        # Lazy — the Face detector (and its model load) is only built
        # the first time FaceController actually activates, not here.
        return _build_detector(model_extension, models_root, face_cfg)

    return FaceController(
        job_id=job_id,
        detector_factory=detector_factory,
        off_timeout_sec=float(trigger_cfg.get("off_timeout_sec", 12)),
        inference_frame_interval=int(trigger_cfg.get("inference_frame_interval", 1)),
    )
