"""JobConfig — the in-memory representation of one entry in jobs.yaml."""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class JobConfig:
    job_id: str
    model_name: str
    model_type: str
    model_extension: str
    master_ip: str
    stream_port: int
    stream_name: str

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @staticmethod
    def from_dict(raw: dict) -> "JobConfig":
        return JobConfig(
            job_id=str(raw["job_id"]),
            model_name=str(raw["model_name"]),
            model_type=str(raw["model_type"]),
            model_extension=str(raw["model_extension"]),
            master_ip=str(raw["master_ip"]),
            stream_port=int(raw["stream_port"]),
            stream_name=str(raw["stream_name"]),
        )

    def config_changed(self, other: "JobConfig") -> bool:
        """True if anything other than job_id differs — used by
        job_watcher.py to classify CHANGED vs UNCHANGED.
        """
        return self.to_dict() != other.to_dict()
