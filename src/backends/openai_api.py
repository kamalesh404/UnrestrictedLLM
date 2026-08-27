"""OpenAI-compatible API backend — Together, OpenRouter, Groq, etc."""

import time
import json
import logging
import requests
from typing import Optional, Generator
from .base import InferenceBackend, InferenceResult, InferenceConfig

logger = logging.getLogger("unrestrictedllm.backend.openai_api")


class OpenAIBackend(InferenceBackend):
    def __init__(self, model_name: str, api_key: str, base_url: str = "https://api.openai.com/v1",
                 **kwargs):
        super().__init__(model_name)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.load()

    def load(self):
        try:
            resp = requests.get(f"{self.base_url}/models", headers=self.headers, timeout=10)
            resp.raise_for_status()
            self._loaded = True
            logger.info(f"Connected to API: {self.base_url} (model: {self.model_name})")
        except requests.ConnectionError:
            raise ConnectionError(f"Cannot connect to {self.base_url}")

    def generate(self, messages: list, config: InferenceConfig) -> InferenceResult:
        start = time.time()
        payload = {
            "model": self.model_name, "messages": messages, "max_tokens": config.max_tokens,
            "temperature": config.temperature, "top_p": config.top_p, "stream": False,
        }
        if config.stop:
            payload["stop"] = config.stop
        resp = requests.post(f"{self.base_url}/chat/completions", json=payload,
                             headers=self.headers, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.time() - start
        text = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        tps = tokens / elapsed if elapsed > 0 else 0
        return InferenceResult(text=text, tokens_generated=tokens, tokens_per_second=tps,
                               finish_reason=data["choices"][0].get("finish_reason", "stop"),
                               model=data.get("model", self.model_name))

    def generate_stream(self, messages: list, config: InferenceConfig) -> Generator[str, None, None]:
        payload = {
            "model": self.model_name, "messages": messages, "max_tokens": config.max_tokens,
            "temperature": config.temperature, "top_p": config.top_p, "stream": True,
        }
        resp = requests.post(f"{self.base_url}/chat/completions", json=payload,
                             headers=self.headers, stream=True, timeout=300)
        for line in resp.iter_lines():
            if line and line.startswith(b"data: "):
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    yield delta["content"]

    def unload(self):
        self._loaded = False

    def info(self) -> dict:
        return {"backend": "openai_api", "model": self.model_name, "base_url": self.base_url, "loaded": self._loaded}
