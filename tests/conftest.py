"""Shared test fixtures."""

import pytest
from pathlib import Path
from src.core.conversation import Conversation
from src.core.config import Config


@pytest.fixture
def config():
    return Config(models_dir=Path("/tmp/test_models"))


@pytest.fixture
def conversation():
    return Conversation(system_prompt="You are a test assistant.")


@pytest.fixture
def sample_messages():
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
    ]
