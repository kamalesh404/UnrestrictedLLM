"""Message formatting and markdown rendering utilities."""

import re
from typing import Optional


def format_tokens(count: int) -> str:
    if count < 1000:
        return f"{count} tokens"
    elif count < 1000000:
        return f"{count/1000:.1f}K tokens"
    return f"{count/1000000:.1f}M tokens"


def format_speed(tokens_per_sec: float) -> str:
    if tokens_per_sec < 1:
        return f"{tokens_per_sec*1000:.0f} ms/token"
    return f"{tokens_per_sec:.1f} tok/s"


def format_size(bytes_val: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} PB"


def strip_markdown(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'```.*?```', '[code block]', text, flags=re.DOTALL)
    return text


def truncate(text: str, max_length: int = 500) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def extract_code_blocks(text: str) -> list[dict]:
    pattern = r'```(\w*)\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return [{"language": lang or "text", "code": code.strip()} for lang, code in matches]
