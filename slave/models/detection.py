"""Detection — a single normalized detection box, and ClassDetection —
the grouped-by-class shape actually published on cam.detection.

Every detector (Person or Face; .pt / .nb / .so) still produces a flat
`List[Detection]` per frame exactly as before — that per-box contract
is unchanged and is what frame_annotator.py draws. What changed is how
that flat list gets published: cam.detection's `telemetry` array is
grouped one entry per class_name (all of a frame's person boxes
together, all of its face boxes together, etc.), each carrying
Master-supplied identity (camera_model_assign_id) and a `count`. See
group_detections() below and processes/detection/detection_result.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from slave.models.job import JobConfig

# Per class_name: (event_type published to Master, extra per-instance
# fields — always present, null until the underlying model/matching
# that fills them in actually exists). Add a new class here (and only
# here) when a new model type starts producing detections.
_CLASS_META: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "person": ("person_count_detected", ("gender",)),
    "face": ("face_recognized", ("name", "authorized", "known", "embedding", "embedding_model")),
    "fire": ("fire_detected", ("density",)),
    "smoke": ("smoke_detected", ()),
    "number_plate": ("number_plate_detected", ("plate_number", "plate_confidence")),
}
_DEFAULT_EVENT_TYPE = "{class_name}_detected"


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 in original frame pixels
    model_name: str  # filename of the model that produced this detection
    # Per-detection values for this class_name's _CLASS_META extra
    # fields (e.g. number_plate's plate_number/plate_confidence) that
    # the detector/worker actually computed, keyed by field name.
    # Empty for every detector that has no such value yet -- those
    # fields stay null via group_detections()'s default fill below.
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = {
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "event": f"{self.class_name}_detected",
            "confidence": float(self.confidence),
            "bbox": [float(v) for v in self.bbox],
            "model_name": self.model_name,
        }
        result.update(self.extra)
        return result


@dataclass(frozen=True)
class DetectionInstance:
    """One physical box within a ClassDetection group."""

    confidence: float
    bbox: Tuple[float, float, float, float]
    extra: Dict[str, Optional[object]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        result: dict = {
            "confidence": float(self.confidence),
            "bbox": [float(v) for v in self.bbox],
        }
        result.update(self.extra)
        return result


@dataclass(frozen=True)
class ClassDetection:
    """One telemetry[] entry — all of a frame's detections for a single
    class_name, grouped together with the identity Master needs to
    attribute them back to the right camera+model assignment.
    """

    camera_model_assign_id: Optional[str]
    class_id: int
    class_name: str
    event_type: str
    instances: Tuple[DetectionInstance, ...]
    tracking_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "camera_model_assign_id": self.camera_model_assign_id,
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "event_type": self.event_type,
            "tracking_id": self.tracking_id,
            "detection": [i.to_dict() for i in self.instances],
            "count": len(self.instances),
        }


def group_detections(
    detections: Sequence[Detection],
    job: JobConfig,
    tracking_id: Optional[str],
    assign_id: Optional[str] = None,
) -> List[ClassDetection]:
    """Group a frame's flat Detection list into one ClassDetection per
    class_name, in the shape cam.detection's `telemetry` array expects.

    camera_model_assign_id per group comes from job.model_assignments
    (Master-supplied, keyed by model_type == class_name) -- unless
    `assign_id` is given, in which case every group uses that one id
    instead. A multi-class model like Safety (boots, no_vest, hardhat,
    ...) needs this: its class names never equal its model_type
    ("safety"), so the per-class lookup would return null for all of
    them. tracking_id
    is the same value for every group in a frame — the frame's saved
    MinIO image id (see detection_result.py), not a real per-object
    tracker.
    """
    by_class: "Dict[str, List[Detection]]" = {}
    order: List[str] = []
    for det in detections:
        if det.class_name not in by_class:
            by_class[det.class_name] = []
            order.append(det.class_name)
        by_class[det.class_name].append(det)

    groups: List[ClassDetection] = []
    for class_name in order:
        class_dets = by_class[class_name]
        event_type, extra_fields = _CLASS_META.get(
            class_name, (_DEFAULT_EVENT_TYPE.format(class_name=class_name), ())
        )
        instances = tuple(
            DetectionInstance(
                confidence=d.confidence,
                bbox=d.bbox,
                extra={**{field_name: None for field_name in extra_fields}, **d.extra},
            )
            for d in class_dets
        )
        groups.append(
            ClassDetection(
                camera_model_assign_id=assign_id if assign_id is not None else job.camera_model_assign_id_for(class_name),
                class_id=class_dets[0].class_id,
                class_name=class_name,
                event_type=event_type,
                instances=instances,
                tracking_id=tracking_id,
            )
        )
    return groups
