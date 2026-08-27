"""llama.cpp backend for GGUF models — CPU and GPU inference."""

import time
import logging
from typing import Optional, Generator
from .base import InferenceBackend, InferenceResult, InferenceConfig

logger = logging.getLogger("unrestrictedllm.backend.llama_cpp")


class LlamaCppBackend(InferenceBackend):
    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1,
                 n_threads: Optional[int] = None, n_batch: int = 512, **kwargs):
        super().__init__(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.n_batch = n_batch
        self._llama = None
        self.load()

    def load(self):
        try:
            from llama_cpp import Llama
            self._llama = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                n_batch=self.n_batch,
                verbose=False,
            )
            self._loaded = True
            logger.info(f"Loaded GGUF model: {self.model_path} (ctx={self.n_ctx}, gpu_layers={self.n_gpu_layers})")
        except ImportError:
            raise ImportError("llama-cpp-python is required: pip install llama-cpp-python")

    def generate(self, messages: list, config: InferenceConfig) -> InferenceResult:
        formatted = self._format_messages(messages)
        start = time.time()
        response = self._llama.create_chat_completion(
            messages=formatted,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repeat_penalty,
            stop=config.stop or None,
            stream=False,
        )
        elapsed = time.time() - start
        text = response["choices"][0]["message"]["content"]
        tokens = response.get("usage", {}).get("completion_tokens", 0)
        tps = tokens / elapsed if elapsed > 0 else 0
        return InferenceResult(text=text, tokens_generated=tokens, tokens_per_second=tps,
                               finish_reason=response["choices"][0].get("finish_reason", "stop"),
                               model=response.get("model", ""))

    def generate_stream(self, messages: list, config: InferenceConfig) -> Generator[str, None, None]:
        formatted = self._format_messages(messages)
        stream = self._llama.create_chat_completion(
            messages=formatted,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repeat_penalty,
            stop=config.stop or None,
            stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                yield delta["content"]

    def unload(self):
        self._llama = None
        self._loaded = False
        logger.info("Unloaded llama.cpp model")

    def info(self) -> dict:
        return {
            "backend": "llama_cpp", "model_path": self.model_path,
            "n_ctx": self.n_ctx, "n_gpu_layers": self.n_gpu_layers,
            "n_threads": self.n_threads, "loaded": self._loaded,
        }

    def _format_messages(self, messages: list) -> list:
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append({"role": role, "content": content})
        return formatted
