"""API routes — OpenAI-compatible endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, list
from ..core.conversation import Conversation
from ..core.model_manager import ModelManager

router = APIRouter()
_manager: Optional[ModelManager] = None


def set_manager(manager: ModelManager):
    global _manager
    _manager = manager


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "mistral-7b-instruct"
    messages: list[ChatMessage]
    max_tokens: int = Field(2048, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0, le=2)
    top_p: float = Field(0.9, ge=0, le=1)
    stream: bool = False
    stop: Optional[list[str]] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict]
    usage: dict


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "unrestrictedllm"


@router.get("/models")
async def list_models():
    if _manager is None:
        raise HTTPException(503, "Model manager not initialized")
    models = _manager.list_models()
    return {"object": "list", "data": [ModelInfo(id=m.name, owned_by="community") for m in models]}


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if _manager is None:
        raise HTTPException(503, "Model manager not initialized")
    if not _manager.is_loaded:
        raise HTTPException(503, "No model loaded. Load a model first.")
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    from ..backends.base import InferenceConfig
    config = InferenceConfig(
        max_tokens=request.max_tokens, temperature=request.temperature,
        top_p=request.top_p, stream=request.stream, stop=request.stop or [],
    )
    result = _manager.current_model.generate(messages, config)
    import time
    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}", created=int(time.time()),
        model=request.model,
        choices=[{"index": 0, "message": {"role": "assistant", "content": result.text},
                  "finish_reason": result.finish_reason}],
        usage={"prompt_tokens": 0, "completion_tokens": result.tokens_generated,
               "total_tokens": result.tokens_generated},
    )


@router.post("/completions")
async def completions(request: ChatCompletionRequest):
    return await chat_completions(request)
