"""FastAPI server — OpenAI-compatible API for UnrestrictedLLM."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ..core.config import Config
from .routes import router


def create_app(config: Config = None) -> FastAPI:
    config = config or Config()
    app = FastAPI(
        title="UnrestrictedLLM API",
        version="0.1.0",
        description="OpenAI-compatible API for uncensored local LLMs",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/v1")
    app.state.config = config

    @app.get("/health")
    async def health():
        return {"status": "healthy", "version": "0.1.0"}

    @app.get("/")
    async def root():
        return {"name": "UnrestrictedLLM", "version": "0.1.0", "docs": "/docs"}

    return app
