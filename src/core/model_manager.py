"""Model manager — download, load, switch, and cache LLM models."""

import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger("unrestrictedllm.model_manager")


@dataclass
class ModelInfo:
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

    def to_dict(self) -> dict:
        return {
            "name": self.name, "repo_id": self.repo_id, "filename": self.filename,
            "size_gb": self.size_gb, "quantization": self.quantization, "backend": self.backend,
            "description": self.description, "context_length": self.context_length,
            "license": self.license, "tags": self.tags,
        }


class ModelManager:
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_model = None
        self._loaded_backend = None
        self._registry: Dict[str, ModelInfo] = {}
        self._load_registry()

    def _load_registry(self):
        registry_file = self.models_dir / "registry.json"
        if registry_file.exists():
            data = json.loads(registry_file.read_text())
            for name, info in data.items():
                self._registry[name] = ModelInfo(**info)

    def _save_registry(self):
        registry_file = self.models_dir / "registry.json"
        data = {name: info.to_dict() for name, info in self._registry.items()}
        registry_file.write_text(json.dumps(data, indent=2))

    def register(self, model: ModelInfo):
        self._registry[model.name] = model
        self._save_registry()
        logger.info(f"Registered model: {model.name}")

    def get_model_path(self, name: str) -> Optional[Path]:
        if name not in self._registry:
            return None
        model = self._registry[name]
        path = self.models_dir / model.filename
        return path if path.exists() else None

    def list_models(self) -> list[ModelInfo]:
        return list(self._registry.values())

    def get_model_info(self, name: str) -> Optional[ModelInfo]:
        return self._registry.get(name)

    def remove_model(self, name: str) -> bool:
        if name not in self._registry:
            return False
        model = self._registry[name]
        path = self.models_dir / model.filename
        if path.exists():
            path.unlink()
            logger.info(f"Deleted model file: {path}")
        del self._registry[name]
        self._save_registry()
        return True

    def load_model(self, name: str, backend: Optional[str] = None, **kwargs):
        if name not in self._registry:
            raise ValueError(f"Model '{name}' not registered")
        model = self._registry[name]
        path = self.get_model_path(name)
        if path is None:
            raise FileNotFoundError(f"Model file not found: {model.filename}")
        backend_name = backend or model.backend
        self._loaded_model = self._init_backend(backend_name, path, model, **kwargs)
        self._loaded_backend = backend_name
        logger.info(f"Loaded model: {name} via {backend_name}")
        return self._loaded_model

    def _init_backend(self, backend_name: str, path: Path, model: ModelInfo, **kwargs):
        if backend_name == "llama_cpp":
            from ..backends.llama_cpp import LlamaCppBackend
            return LlamaCppBackend(str(path), n_ctx=model.context_length, **kwargs)
        elif backend_name == "transformers":
            from ..backends.transformers import TransformersBackend
            return TransformersBackend(str(path), **kwargs)
        elif backend_name == "ollama":
            from ..backends.ollama import OllamaBackend
            return OllamaBackend(model.repo_id, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend_name}")

    @property
    def is_loaded(self) -> bool:
        return self._loaded_model is not None

    @property
    def current_model(self):
        return self._loaded_model
