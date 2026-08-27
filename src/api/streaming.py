"""SSE streaming for chat responses."""

import json
import asyncio
from typing import AsyncGenerator


async def stream_response(generator, model: str = "") -> AsyncGenerator[str, None]:
    """Convert a sync generator to async SSE stream."""
    loop = asyncio.get_event_loop()
    for chunk in generator:
        data = {
            "id": f"chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(data)}\n\n"
        await loop.run_in_executor(None, lambda: None)
    done = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    yield f"data: {json.dumps(done)}\n\n"
    yield "data: [DONE]\n\n"
