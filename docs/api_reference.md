# API Reference

## CLI

### pull
Download a model.
```bash
unrestricted pull <model-name>
```

### run
Interactive chat mode.
```bash
unrestricted run [--model NAME] [--backend BACKEND]
```

### serve
Start OpenAI-compatible API server.
```bash
unrestricted serve [--host HOST] [--port PORT] [--model NAME]
```

### models
List available models.
```bash
unrestricted models
```

### info
Show model details.
```bash
unrestricted info <model-name>
```

## Python API

### Conversation

```python
from unrestrictedllm.core.conversation import Conversation

conv = Conversation(system_prompt="You are helpful.")
conv.add_user("Hello")
conv.add_assistant("Hi!")
messages = conv.get_messages()
```

### ModelManager

```python
from unrestrictedllm.core.model_manager import ModelManager

manager = ModelManager(models_dir)
manager.load_model("dolphin-2.2.1-mistral-7b", backend="llama_cpp")
result = manager.current_model.generate(messages, config)
```

### InferenceConfig

```python
from unrestrictedllm.backends.base import InferenceConfig

config = InferenceConfig(
    max_tokens=2048,
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    repeat_penalty=1.1,
    stream=False,
)
```

## HTTP API

### POST /v1/chat/completions

```json
{
  "model": "dolphin-2.2.1-mistral-7b",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 512,
  "temperature": 0.7,
  "stream": false
}
```

### GET /v1/models

Lists available models.

### GET /health

Health check endpoint.
