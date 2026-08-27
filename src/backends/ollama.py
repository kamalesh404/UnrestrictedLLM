"""Ollama API backend — connect to a local or remote Ollama server."""

import time
import json
import logging
import requests
from typing import Optional, Generator
from .base import InferenceBackend, InferenceResult, InferenceConfig

logger = logging.getLogger("unrestrictedllm.backend.ollama")


class OllamaBackend(InferenceBackend):
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(model_name)
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.load()

    def load(self):
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            available = [m["name"] for m in resp.json().get("models", [])]
            if self.model_name not in available:
                logger.warning(f"Model '{self.model_name}' not found locally. Pulling...")
                self._pull_model()
            self._loaded = True
            logger.info(f"Connected to Ollama: {self.model_name}")
        except requests.ConnectionError:
            raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}")

    def _pull_model(self):
        resp = requests.post(f"{self.base_url}/api/pull", json={"name": self.model_name}, stream=True, timeout=300)
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                status = data.get("status", "")
                if "error" in data:
                    raise RuntimeError(f"Ollama pull error: {data['error']}")
                logger.info(f"Pulling: {status}")

    def generate(self, messages: list, config: InferenceConfig) -> InferenceResult:
        start = time.time()
        resp = requests.post(f"{self.base_url}/api/chat", json={
            "model": self.model_name, "messages": messages, "stream": False,
            "options": {
                "num_predict": config.max_tokens, "temperature": config.temperature,
                "top_p": config.top_p, "top_k": config.top_k,
                "repeat_penalty": config.repeat_penalty,
            },
        }, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.time() - start
        text = data["message"]["content"]
        tokens = data.get("eval_count", 0)
        tps = tokens / elapsed if elapsed > 0 else 0
        return InferenceResult(text=text, tokens_generated=tokens, tokens_per_second=tps,
                               finish_reason="stop", model=self.model_name)

    def generate_stream(self, messages: list, config: InferenceConfig) -> Generator[str, None, None]:
        resp = requests.post(f"{self.base_url}/api/chat", json={
            "model": self.model_name, "messages": messages, "stream": True,
            "options": {
                "num_predict": config.max_tokens, "temperature": config.temperature,
                "top_p": config.top_p, "repeat_penalty": config.repeat_penalty,
            },
        }, stream=True, timeout=300)
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                if "message" in data:
                    yield data["message"].get("content", "")

    def unload(self):
        self._loaded = False
        logger.info(f"Disconnected from Ollama: {self.model_name}")

    def info(self) -> dict:
        return {"backend": "ollama", "model": self.model_name, "base_url": self.base_url, "loaded": self._loaded}
