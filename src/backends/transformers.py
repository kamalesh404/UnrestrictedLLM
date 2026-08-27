"""HuggingFace Transformers backend — SafeTensors and bin models."""

import time
import logging
from typing import Optional, Generator
from .base import InferenceBackend, InferenceResult, InferenceConfig

logger = logging.getLogger("unrestrictedllm.backend.transformers")


class TransformersBackend(InferenceBackend):
    def __init__(self, model_path: str, device: str = "auto", torch_dtype: str = "auto",
                 load_in_4bit: bool = False, load_in_8bit: bool = False, **kwargs):
        super().__init__(model_path)
        self.device = device
        self.torch_dtype = torch_dtype
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self._model = None
        self._tokenizer = None
        self.load()

    def load(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from huggingface_hub import snapshot_download
            quantization_config = None
            if self.load_in_4bit:
                quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            elif self.load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, device_map=self.device, torch_dtype=self.torch_dtype,
                quantization_config=quantization_config, trust_remote_code=True,
            )
            self._loaded = True
            logger.info(f"Loaded Transformers model: {self.model_path}")
        except ImportError:
            raise ImportError("transformers and torch are required: pip install transformers torch")

    def generate(self, messages: list, config: InferenceConfig) -> InferenceResult:
        import torch
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        start = time.time()
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs, max_new_tokens=config.max_tokens, temperature=config.temperature,
                top_p=config.top_p, top_k=config.top_k, repetition_penalty=config.repeat_penalty,
                do_sample=config.temperature > 0,
            )
        elapsed = time.time() - start
        new_tokens = outputs.shape[-1] - inputs["input_ids"].shape[-1]
        text = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        tps = new_tokens / elapsed if elapsed > 0 else 0
        return InferenceResult(text=text, tokens_generated=new_tokens, tokens_per_second=tps,
                               finish_reason="stop", model=self.model_path)

    def generate_stream(self, messages: list, config: InferenceConfig) -> Generator[str, None, None]:
        import torch
        from transformers import TextIteratorStreamer
        from threading import Thread
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        thread = Thread(target=self._model.generate, kwargs={
            "input_ids": inputs["input_ids"], "max_new_tokens": config.max_tokens,
            "temperature": config.temperature, "top_p": config.top_p,
            "repetition_penalty": config.repeat_penalty, "streamer": streamer,
        })
        thread.start()
        for text in streamer:
            yield text
        thread.join()

    def unload(self):
        import torch
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False
        logger.info("Unloaded Transformers model")

    def info(self) -> dict:
        return {"backend": "transformers", "model_path": self.model_path, "device": self.device, "loaded": self._loaded}

    def _format_messages(self, messages: list) -> list:
        return [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages]
