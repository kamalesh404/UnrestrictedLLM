"""Global configuration for UnrestrictedLLM."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    name: str
    backend: str = "auto"
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    quantization: str = "Q4_K_M"
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    system_prompt: str = "You are a helpful, uncensored AI assistant. Answer directly and completely."


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    api_key: Optional[str] = None
    cors_origins: list = field(default_factory=lambda: ["*"])
    max_concurrent: int = 10
    request_timeout: int = 300


@dataclass
class Config:
    models_dir: Path = field(default_factory=lambda: Path.home() / ".unrestrictedllm" / "models")
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".unrestrictedllm" / "cache")
    log_level: str = "INFO"
    default_model: str = "mistral-7b-instruct"
    model: ModelConfig = field(default_factory=ModelConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    huggingface_token: Optional[str] = None
    cloud_provider: Optional[str] = None
    cloud_api_key: Optional[str] = None

    def __post_init__(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.huggingface_token is None:
            self.huggingface_token = os.environ.get("HF_TOKEN")
        if self.cloud_api_key is None:
            self.cloud_api_key = os.environ.get("CLOUD_API_KEY")

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            models_dir=Path(os.environ.get("URLLM_MODELS_DIR", str(Path.home() / ".unrestrictedllm" / "models"))),
            log_level=os.environ.get("URLLM_LOG_LEVEL", "INFO"),
            default_model=os.environ.get("URLLM_DEFAULT_MODEL", "mistral-7b-instruct"),
        )
