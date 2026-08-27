# Cloud Deployment

Deploy UnrestrictedLLM to any cloud provider.

## Modal (Serverless GPU)

```python
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
    result = manager.current_model.generate(
        messages=[{"role": "user", "content": prompt}]
    )
    return {"response": result.text}
```

## RunPod

```bash
# Create pod with GPU
runpodctl create pod --gpu-type RTX_4090 --image unrestrictedllm:latest

# SSH in and run
unrestricted serve --port 8080
```

## Lambda Labs

```bash
# Rent a GPU instance
lambda deploy --instance-type gpu-l4 --image unrestrictedllm:latest
```

## Own Server (Hetzner, DigitalOcean, etc.)

```bash
# Install
curl -sSL https://unrestrictedllm.sh | bash

# Run as service
unrestricted serve --port 8080 &

# Use systemd for persistence
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace token (for gated models) |
| `CLOUD_API_KEY` | Cloud API key |
| `URLLM_MODELS_DIR` | Custom models directory |
| `URLLM_DEFAULT_MODEL` | Default model name |
| `OPENAI_API_KEY` | For OpenAI-compatible cloud backends |
