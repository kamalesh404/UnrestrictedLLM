"""Chat conversation management with history and message formatting."""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: Role
    content: str
    timestamp: float = field(default_factory=time.time)
    tokens: int = 0

    def to_dict(self) -> dict:
        return {"role": self.role.value, "content": self.content}

    def __len__(self) -> int:
        return len(self.content)


class Conversation:
    def __init__(self, system_prompt: str = "", max_history: int = 50):
        self.messages: list[Message] = []
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.created_at = time.time()

    def add_user(self, content: str) -> Message:
        msg = Message(role=Role.USER, content=content)
        self.messages.append(msg)
        self._trim()
        return msg

    def add_assistant(self, content: str) -> Message:
        msg = Message(role=Role.ASSISTANT, content=content)
        self.messages.append(msg)
        self._trim()
        return msg

    def get_messages(self, include_system: bool = True) -> list[dict]:
        result = []
        if include_system and self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})
        result.extend([m.to_dict() for m in self.messages])
        return result

    def format_for_llama_cpp(self) -> str:
        parts = []
        if self.system_prompt:
            parts.append(f"[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n")
        for i, msg in enumerate(self.messages):
            if msg.role == Role.USER:
                if i == 0 and self.system_prompt:
                    parts[-1] += f"{msg.content} [/INST]"
                else:
                    parts.append(f"[INST] {msg.content} [/INST]")
            elif msg.role == Role.ASSISTANT:
                parts.append(f" {msg.content} </s>")
        return "".join(parts)

    def format_for_chatml(self) -> list[dict]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        for msg in self.messages:
            messages.append(msg.to_dict())
        return messages

    def export_json(self) -> str:
        return json.dumps({
            "system_prompt": self.system_prompt,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at,
        }, indent=2)

    def clear(self):
        self.messages.clear()
        self.created_at = time.time()

    def _trim(self):
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:
        return f"Conversation(messages={len(self.messages)}, system={bool(self.system_prompt)})"
