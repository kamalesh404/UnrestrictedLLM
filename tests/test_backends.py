"""Tests for inference backends."""

import pytest
from src.backends.base import InferenceConfig, InferenceResult


def test_inference_config_defaults():
    config = InferenceConfig()
    assert config.max_tokens == 2048
    assert config.temperature == 0.7
    assert config.top_p == 0.9


def test_inference_result():
    result = InferenceResult(text="Hello", tokens_generated=5, tokens_per_second=10.0)
    assert result.text == "Hello"
    assert result.tokens_generated == 5
