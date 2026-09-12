"""Registry of (model_type, model_extension) -> worker builder.

The registry is deliberately dumb: it just maps supported combinations
to the callable that builds the right worker class. `person` and
standalone `face` are registered today; fire/fight are rejected until
implemented (CLAUDE.md §7, §22 — note Face going standalone here is a
deliberate, explicitly-requested exception to CLAUDE.md's "Person
only" scope, not a silent expansion).
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

WorkerBuilder = Callable[..., object]

SUPPORTED_MODEL_TYPES = ("person", "face", "safety")
SUPPORTED_EXTENSIONS = (".pt", ".nb", ".so")


class ModelNotSupportedError(ValueError): 
    pass


class ModelRegistry: # model registry is a class that maintains a mapping of supported (model_type, model_extension) combinations to their corresponding worker builders. It provides methods to register new worker builders and retrieve them based on the model type and extension. If an unsupported combination is requested, it raises a ModelNotSupportedError.
    def __init__(self) -> None:
        self._builders: Dict[Tuple[str, str], WorkerBuilder] = {} # 

    def register(self, model_type: str, model_extension: str, builder: WorkerBuilder) -> None:
        self._builders[(model_type, model_extension)] = builder

    def get(self, model_type: str, model_extension: str) -> WorkerBuilder:
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ModelNotSupportedError(
                f"model_type='{model_type}' is not implemented yet "
                f"(supported: {SUPPORTED_MODEL_TYPES})"
            )
        if model_extension not in SUPPORTED_EXTENSIONS:
            raise ModelNotSupportedError(
                f"model_extension='{model_extension}' is not supported "
                f"(supported: {SUPPORTED_EXTENSIONS})"
            )
        key = (model_type, model_extension)
        if key not in self._builders:
            raise ModelNotSupportedError(f"no worker registered for {key}")
        return self._builders[key]


def build_default_registry() -> ModelRegistry: # build_default_registry is a function that creates and returns a ModelRegistry instance pre-populated with the current worker builders for the "person"/"face" model types and their supported extensions (".pt", ".nb", ".so"). It imports the necessary worker classes and registers them with the registry, allowing the system to retrieve the appropriate worker builder based on the model type and extension.
    """Registry wired with the current Person + standalone Face worker builders."""
    from slave.processes.worker.models.person.pt.person_pt_worker import PersonPtWorker
    from slave.processes.worker.models.person.nb.person_nb_worker import PersonNbWorker
    from slave.processes.worker.models.person.so.person_so_worker import PersonSoWorker
    from slave.processes.worker.models.face.pt.face_pt_worker import FacePtWorker
    from slave.processes.worker.models.face.nb.face_nb_worker import FaceNbWorker
    from slave.processes.worker.models.face.so.face_so_worker import FaceSoWorker
    from slave.processes.worker.models.safety.pt.safety_pt_worker import SafetyPtWorker
    from slave.processes.worker.models.safety.nb.safety_nb_worker import SafetyNbWorker
    from slave.processes.worker.models.safety.so.safety_so_worker import SafetySoWorker

    registry = ModelRegistry() # creates a new ModelRegistry instance
    registry.register("person", ".pt", PersonPtWorker) # model_type is "person", model_extension is ".pt", and the corresponding worker builder is PersonPtWorker. This registers the PersonPtWorker class as the worker builder for the combination of model_type "person" and model_extension ".pt".
    registry.register("person", ".nb", PersonNbWorker) # model_type is "person", model_extension is ".nb", and the corresponding worker builder is PersonNbWorker. This registers the PersonNbWorker class as the worker builder for the combination of model_type "person" and model_extension ".nb".
    registry.register("person", ".so", PersonSoWorker) # model_type is "person", model_extension is ".so", and the corresponding worker builder is PersonSoWorker. This is the library support for the person.nb model, which is a shared object file. This registers the PersonSoWorker class as the worker builder for the combination of model_type "person" and model_extension ".so".
    registry.register("face", ".pt", FacePtWorker)   # standalone Face job (no Person job required) — same trigger-free Face detector used by face_controller.py, just running as its own job instead of person-chained.
    registry.register("face", ".nb", FaceNbWorker)
    registry.register("face", ".so", FaceSoWorker)
    registry.register("safety", ".pt", SafetyPtWorker)   # standalone Safety job — 9-class PPE-compliance detector, also face-controller-attached (Face auto-triggers off Safety's own "person" class, see worker_factory.py).
    registry.register("safety", ".nb", SafetyNbWorker)
    registry.register("safety", ".so", SafetySoWorker)
    return registry # returns the fully constructed ModelRegistry instance with the registered worker builders for the supported (model_type, model_extension) combinations.
