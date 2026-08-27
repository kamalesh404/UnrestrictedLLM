"""Base class for inference backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Generator


@dataclass
class InferenceResult:
    text: str
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    finish_reason: str = "stop"
    model: str = ""


@dataclass
class InferenceConfig:
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop: list = field(default_factory=list)
    stream: bool = False


class InferenceBackend(ABC):
    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    def generate(self, messages: list, config: InferenceConfig) -> InferenceResult:
        """Generate a response from messages."""
        pass

    @abstractmethod
    def generate_stream(self, messages: list, config: InferenceConfig) -> Generator[str, None, None]:
        """Stream tokens as they are generated."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the model from memory."""
        pass

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def info(self) -> dict:
        """Return backend and model info."""
        pass
