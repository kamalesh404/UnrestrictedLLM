"""Tokenizer wrapper supporting HuggingFace and GGUF tokenizers."""

from typing import Optional, List
from dataclasses import dataclass


@dataclass
class TokenCount:
    input_tokens: int
    output_tokens: int = 0
    total: int = 0

    def __post_init__(self):
        self.total = self.input_tokens + self.output_tokens


class Tokenizer:
    def __init__(self, backend: str = "auto", model_path: Optional[str] = None):
        self.backend = backend
        self.model_path = model_path
        self._tokenizer = None
        self._init_tokenizer()

    def _init_tokenizer(self):
        if self.backend == "llama_cpp" and self.model_path:
            try:
                from llama_cpp import Llama
                llama = Llama(self.model_path, n_ctx=0, n_batch=0)
                self._tokenizer = llama
                self.backend = "llama_cpp"
            except ImportError:
                self._init_hf_tokenizer()
        else:
            self._init_hf_tokenizer()

    def _init_hf_tokenizer(self):
        try:
            from transformers import AutoTokenizer
            if self.model_path:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                self.backend = "huggingface"
        except (ImportError, Exception):
            self._tokenizer = None
            self.backend = "fallback"

    def encode(self, text: str) -> List[int]:
        if self._tokenizer is None:
            return self._fallback_encode(text)
        if self.backend == "llama_cpp":
            return self._tokenizer.tokenize(text.encode("utf-8"))
        elif self.backend == "huggingface":
            return self._tokenizer.encode(text)
        return self._fallback_encode(text)

    def decode(self, tokens: List[int]) -> str:
        if self._tokenizer is None:
            return self._fallback_decode(tokens)
        if self.backend == "llama_cpp":
            return self._tokenizer.detokenize(tokens).decode("utf-8", errors="replace")
        elif self.backend == "huggingface":
            return self._tokenizer.decode(tokens, skip_special_tokens=True)
        return self._fallback_decode(tokens)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def count_messages(self, messages: list) -> TokenCount:
        total = sum(self.count_tokens(m.get("content", "")) for m in messages)
        return TokenCount(input_tokens=total)

    def _fallback_encode(self, text: str) -> List[int]:
        return list(range(len(text)))

    def _fallback_decode(self, tokens: List[int]) -> str:
        return "".join(chr(t % 128) for t in tokens)

    @property
    def vocab_size(self) -> int:
        if self._tokenizer is None:
            return 32000
        if hasattr(self._tokenizer, "vocab_size"):
            return self._tokenizer.vocab_size
        return 32000

    @property
    def eos_token(self) -> str:
        if self._tokenizer is None:
            return "</s>"
        if hasattr(self._tokenizer, "eos_token"):
            return self._tokenizer.eos_token
        return "</s>"

    @property
    def pad_token(self) -> str:
        if self._tokenizer is None:
            return "<pad>"
        if hasattr(self._tokenizer, "pad_token") and self._tokenizer.pad_token:
            return self._tokenizer.pad_token
        return self.eos_token
