# Getting Started

## Installation

```bash
# Clone
git clone https://github.com/kamalesh404/UnrestrictedLLM.git
cd UnrestrictedLLM

# Install
pip install -e .

# Or install specific extras
pip install -e ".[llama_cpp]"   # GGUF support
pip install -e ".[transformers]" # HF models
pip install -e ".[ui]"          # Web UI
pip install -e ".[cloud]"       # Cloud deployment
```

## Download a Model

```bash
# List available models
unrestricted models

# Download Dolphin 7B (uncensored)
unrestricted pull dolphin-2.2.1-mistral-7b

# Check model info
unrestricted info dolphin-2.2.1-mistral-7b
```

## Start Chatting

```bash
# Interactive chat
unrestricted run --model dolphin-2.2.1-mistral-7b

# With Transformers backend
unrestricted run --model openhermes-7b --backend transformers

# Start API server
unrestricted serve --model dolphin-2.2.1-mistral-7b --port 8080
```

## Web UI

```bash
# Start the Gradio UI
python -m src.ui.app

# Then open http://localhost:7860
```

## System Requirements

- **CPU-only:** 8GB+ RAM for 7B models (Q4 quantization)
- **With GPU (recommended):** 6GB+ VRAM for 7B models
- **Storage:** ~5GB per 7B model
- **Works on:** Windows, macOS (M1/M2), Linux
