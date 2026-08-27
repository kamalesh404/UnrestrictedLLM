"""Model registry — track available models, their metadata, and status."""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class ModelStatus(str, Enum):
    AVAILABLE = "available"
    DOWNLOADING = "downloading"
    LOADED = "loaded"
    ERROR = "error"


@dataclass
class ModelEntry:
    name: str
    repo_id: str
    filename: str
    size_gb: float
    quantization: str
    backend: str
    description: str = ""
    context_length: int = 4096
    license: str = ""
    tags: list = field(default_factory=list)
    status: ModelStatus = ModelStatus.AVAILABLE

    def to_dict(self) -> dict:
        return {
            "name": self.name, "repo_id": self.repo_id, "filename": self.filename,
            "size_gb": self.size_gb, "quantization": self.quantization, "backend": self.backend,
            "description": self.description, "context_length": self.context_length,
            "license": self.license, "tags": self.tags, "status": self.status.value,
        }

    def display(self) -> str:
        tags = ", ".join(self.tags[:3])
        return f"{self.name:30s} | {self.size_gb:5.1f}GB | {self.quantization:8s} | {tags}"


class ModelRegistry:
    def __init__(self):
        self._models: dict[str, ModelEntry] = {}

    def register(self, entry: ModelEntry):
        self._models[entry.name] = entry

    def get(self, name: str) -> Optional[ModelEntry]:
        return self._models.get(name)

    def list_all(self) -> list[ModelEntry]:
        return list(self._models.values())

    def list_by_tag(self, tag: str) -> list[ModelEntry]:
        return [m for m in self._models.values() if tag in m.tags]

    def list_by_backend(self, backend: str) -> list[ModelEntry]:
        return [m for m in self._models.values() if m.backend == backend]

    def search(self, query: str) -> list[ModelEntry]:
        query = query.lower()
        return [m for m in self._models.values()
                if query in m.name.lower() or query in m.description.lower()
                or any(query in t for t in m.tags)]

    def remove(self, name: str) -> bool:
        if name in self._models:
            del self._models[name]
            return True
        return False

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, name: str) -> bool:
        return name in self._models
