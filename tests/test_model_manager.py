"""Tests for model manager."""

import pytest
from pathlib import Path
from src.core.model_manager import ModelManager, ModelInfo


def test_model_manager_init(tmp_path):
    manager = ModelManager(tmp_path)
    assert manager.models_dir == tmp_path
    assert not manager.is_loaded


def test_model_manager_register(tmp_path):
    manager = ModelManager(tmp_path)
    model = ModelInfo(name="test", repo_id="test/repo", filename="test.gguf",
                      size_gb=1.0, quantization="Q4", backend="llama_cpp")
    manager.register(model)
    assert len(manager.list_models()) == 1
    assert manager.get_model_info("test") is not None


def test_model_manager_remove(tmp_path):
    manager = ModelManager(tmp_path)
    model = ModelInfo(name="test", repo_id="test/repo", filename="test.gguf",
                      size_gb=1.0, quantization="Q4", backend="llama_cpp")
    manager.register(model)
    assert manager.remove_model("test")
    assert len(manager.list_models()) == 0


def test_model_manager_get_missing(tmp_path):
    manager = ModelManager(tmp_path)
    assert manager.get_model_info("nonexistent") is None
    assert manager.get_model_path("nonexistent") is None
