"""Registry of (model_type, model_extension) -> worker builder.

The registry is deliberately dumb: it just maps supported combinations
to the callable that builds the right worker class. Only `person` is
registered today; face/fire/fight are rejected until implemented
(CLAUDE.md §7, §22).
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

WorkerBuilder = Callable[..., object]

SUPPORTED_MODEL_TYPES = ("person",)
SUPPORTED_EXTENSIONS = (".pt", ".nb", ".so")


class ModelNotSupportedError(ValueError):
    pass


class ModelRegistry:
    def __init__(self) -> None:
        self._builders: Dict[Tuple[str, str], WorkerBuilder] = {}

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


def build_default_registry() -> ModelRegistry:
    """Registry wired with the current Person worker builders."""
    from slave.processes.worker.models.person.pt.person_pt_worker import PersonPtWorker
    from slave.processes.worker.models.person.nb.person_nb_worker import PersonNbWorker
    from slave.processes.worker.models.person.so.person_so_worker import PersonSoWorker

    registry = ModelRegistry()
    registry.register("person", ".pt", PersonPtWorker)
    registry.register("person", ".nb", PersonNbWorker)
    registry.register("person", ".so", PersonSoWorker)
    return registry
