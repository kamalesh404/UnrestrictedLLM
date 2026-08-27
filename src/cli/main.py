"""CLI entry point — UnrestrictedLLM command-line interface."""

import click
import sys
from pathlib import Path
from ..core.config import Config


@click.group()
@click.option("--models-dir", type=click.Path(), default=None, help="Models directory")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def cli(ctx, models_dir, verbose):
    """UnrestrictedLLM — Run uncensored AI models locally or on cloud."""
    ctx.ensure_object(dict)
    config = Config.from_env()
    if models_dir:
        config.models_dir = Path(models_dir)
    if verbose:
        config.log_level = "DEBUG"
    ctx.obj["config"] = config


@cli.command()
@click.argument("name")
@click.option("--backend", "-b", default=None, help="Backend (llama_cpp, transformers, ollama)")
@click.pass_context
def pull(ctx, name, backend):
    """Download a model from HuggingFace."""
    from ..models.downloader import ModelDownloader
    from ..models.uncensored import UNCENSORED_MODELS
    config = ctx.obj["config"]
    model_entry = next((m for m in UNCENSORED_MODELS if m.name == name), None)
    if not model_entry:
        click.echo(f"Unknown model: {name}")
        click.echo("Available models:")
        for m in UNCENSORED_MODELS:
            click.echo(f"  {m.display()}")
        return
    downloader = ModelDownloader(config.models_dir)
    click.echo(f"Downloading {name}...")
    path = downloader.download_model(model_entry.repo_id, model_entry.filename)
    click.echo(f"Downloaded to: {path}")


@cli.command()
@click.option("--model", "-m", default=None, help="Model to load")
@click.option("--backend", "-b", default=None, help="Backend to use")
@click.pass_context
def run(ctx, model, backend):
    """Interactive chat mode."""
    from ..core.model_manager import ModelManager
    from ..core.conversation import Conversation
    config = ctx.obj["config"]
    manager = ModelManager(config.models_dir)
    model_name = model or config.default_model
    click.echo(f"Loading model: {model_name}...")
    try:
        loaded = manager.load_model(model_name, backend)
    except Exception as e:
        click.echo(f"Error loading model: {e}")
        return
    conv = Conversation(system_prompt=config.model.system_prompt)
    click.echo("Chat started. Type 'quit' to exit, 'clear' to reset.\n")
    while True:
        try:
            user_input = click.prompt("You", prompt_suffix=": ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() == "clear":
            conv.clear()
            click.echo("Conversation cleared.")
            continue
        conv.add_user(user_input)
        messages = conv.get_messages()
        from ..backends.base import InferenceConfig
        inf_config = InferenceConfig(max_tokens=config.model.max_tokens, temperature=config.model.temperature)
        result = loaded.generate(messages, inf_config)
        conv.add_assistant(result.text)
        click.echo(f"\nAssistant: {result.text}\n")
        click.echo(f"  [{result.tokens_generated} tokens, {result.tokens_per_second:.1f} tok/s]\n")


@cli.command()
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", "-p", default=8080, type=int, help="Server port")
@click.option("--model", "-m", default=None, help="Model to load")
@click.pass_context
def serve(ctx, host, port, model):
    """Start the OpenAI-compatible API server."""
    import uvicorn
    from ..core.model_manager import ModelManager
    from ..api.server import create_app
    from ..api.routes import set_manager
    config = ctx.obj["config"]
    manager = ModelManager(config.models_dir)
    set_manager(manager)
    model_name = model or config.default_model
    click.echo(f"Loading model: {model_name}...")
    try:
        manager.load_model(model_name)
    except Exception as e:
        click.echo(f"Error: {e}")
        return
    app = create_app(config)
    click.echo(f"Starting server at http://{host}:{port}")
    click.echo(f"API docs at http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port)


@cli.command()
@click.pass_context
def models(ctx):
    """List available models."""
    from ..models.uncensored import UNCENSORED_MODELS
    click.echo(f"{'Name':30s} | {'Size':>5s} | {'Quant':8s} | Tags")
    click.echo("-" * 80)
    for m in UNCENSORED_MODELS:
        click.echo(m.display())


@cli.command()
@click.argument("name")
@click.pass_context
def info(ctx, name):
    """Show model details."""
    from ..models.uncensored import UNCENSORED_MODELS
    model = next((m for m in UNCENSORED_MODELS if m.name == name), None)
    if not model:
        click.echo(f"Model not found: {name}")
        return
    click.echo(f"Name:        {model.name}")
    click.echo(f"Repo:        {model.repo_id}")
    click.echo(f"File:        {model.filename}")
    click.echo(f"Size:        {model.size_gb} GB")
    click.echo(f"Quantization:{model.quantization}")
    click.echo(f"Backend:     {model.backend}")
    click.echo(f"Context:     {model.context_length} tokens")
    click.echo(f"License:     {model.license}")
    click.echo(f"Description: {model.description}")
    click.echo(f"Tags:        {', '.join(model.tags)}")


if __name__ == "__main__":
    cli()
