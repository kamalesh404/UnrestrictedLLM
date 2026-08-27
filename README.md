<div align="center">

# 🔥 UnrestrictedLLM

**Run uncensored open-source LLMs — locally or on cloud**

[![License: MIT](https://img.shields.io/badge/License-MIT-FF4500?style=for-the-badge&logo=mit&logoColor=white)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![CLI](https://img.shields.io/badge/CLI-Click-000000?style=for-the-badge&logo=click&logoColor=white)]()
[![API](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)]()
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](Dockerfile)

**Full control over your AI. No filters. No censorship. Your data, your model, your rules.**

</div>

## What is UnrestrictedLLM?

UnrestrictedLLM is a **privacy-first tool** for running uncensored open-source language models on **your own hardware** or **any cloud provider**. It gives you:

- 🔒 **Full control** — your prompts and data never leave your machine
- 🧠 **Uncensored models** — curated list of community fine-tunes (Dolphin, Airoboros, OpenHermes, etc.)
- 🚀 **Multiple backends** — llama.cpp, Transformers, Ollama, or any OpenAI-compatible API
- 💻 **CLI + Web UI + API** — use it the way you want
- 🌐 **Cloud deploy** — RunPod, Modal, Lambda, or your own server

> **Note:** "Uncensored" refers to open-weight models without restrictive fine-tuning. Use responsibly and in accordance with local laws.

## Quick Start

```bash
# Install
pip install -e .

# Download a model (Dolphin 7B — fully uncensored)
unrestricted pull dolphin-2.2.1-mistral-7b

# Start chatting
unrestricted run --model dolphin-2.2.1-mistral-7b

# Or start the API server
unrestricted serve --model dolphin-2.2.1-mistral-7b --port 8080
```

## Available Models

### 🧑‍💻 Best Coding Models (for 4GB VRAM)

| Model | Size | Description |
|-------|------|-------------|
| **qwen2.5-coder-7b** ⭐ | 4.7GB | Best coding model that fits 4GB. Top-tier code gen |
| **qwen2.5-coder-7b-q3** | 3.7GB | Fully fits 4GB VRAM, slightly lower quality |
| **qwen2.5-coder-3b** | 2.0GB | Tiny & fast, surprisingly capable |
| **deepseek-coder-6.7b** | 4.1GB | Excellent code generation and completion |
| **codellama-7b** | 4.1GB | Meta's reliable coding model |

### 🧠 General Uncensored Models

| Model | Size | Description | Backend |
|-------|------|-------------|---------|
| **dolphin-2.2.1-mistral-7b** | 4.4GB | Fully uncensored, no alignment | llama.cpp |
| **airoboros-mistral-7b** | 4.4GB | Uncensored, roleplay & creative writing | llama.cpp |
| **openhermes-7b** | 4.4GB | Excellent for coding and reasoning | llama.cpp |
| **mistral-7b-instruct** | 4.4GB | Fast, capable, uncensored fine-tune | llama.cpp |
| **llama-3-8b-instruct** | 4.9GB | Meta's latest, highly capable | llama.cpp |
| **solar-10.7b-instruct** | 6.5GB | Multilingual, less censored | llama.cpp |

## Architecture

```
User → CLI / Web UI / API
                 │
                 ▼
        ┌─────────────────┐
        │  Conversation    │  Chat history, system
        │  Manager         │  prompts, formatting
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │   Model Manager  │  Load, switch, cache
        └────────┬────────┘
                 │
        ┌────────┴─────────┐
        │     Backends      │
        ├──────────────────┤
        │  llama.cpp (GGUF) │ ← Local CPU/GPU
        │  Transformers     │ ← Local SafeTensors
        │  Ollama           │ ← Local Ollama server
        │  OpenAI API       │ ← Cloud (Together, etc.)
        └──────────────────┘
```

## Backends

| Backend | Use Case | Command |
|---------|----------|---------|
| **llama.cpp** | Best local performance (CPU/GPU) | `--backend llama_cpp` |
| **Transformers** | Full HuggingFace model access | `--backend transformers` |
| **Ollama** | Use Ollama's model library | `--backend ollama` |
| **OpenAI API** | Cloud inference (Together, Groq) | env `OPENAI_API_KEY` |

### Python API

```python
from unrestrictedllm import Config
from unrestrictedllm.core.manager import ModelManager

config = Config()
manager = ModelManager(config.models_dir)
manager.load_model("dolphin-2.2.1-mistral-7b")

result = manager.current_model.generate(
    messages=[{"role": "user", "content": "Explain quantum computing simply"}],
    config=InferenceConfig(max_tokens=512, temperature=0.7),
)
print(result.text)
```

## API (OpenAI-compatible)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dolphin-2.2.1-mistral-7b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 512
  }'
```

## Cloud Deployment

### Modal (serverless)

```python
# deploy.py
from modal import Image, App, web_endpoint

image = Image.debian_slim().pip_install("unrestrictedllm")

app = App("unrestrictedllm")

@app.function(image=image, gpu="T4")
@web_endpoint()
def chat(prompt: str):
    from unrestrictedllm import Config
    from unrestrictedllm.core.manager import ModelManager
    manager = ModelManager(Config().models_dir)
    manager.load_model("dolphin-2.2.1-mistral-7b")
    result = manager.current_model.generate(messages=[{"role": "user", "content": prompt}])
    return {"response": result.text}
```

### RunPod

```bash
# Pay-as-you-go GPU rental
runpodctl create pod --gpu-type RTX_4090 --image unrestrictedllm:latest
```

## Star History

⭐ **Support the project** — star this repo to help more people discover local, uncensored AI!

## License

MIT License — see [LICENSE](LICENSE)

<div align="center">
**Built with ❤️ by [Kamalesh](https://github.com/kamalesh404)**
</div>
