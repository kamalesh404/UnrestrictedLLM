"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.server import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "UnrestrictedLLM" in response.json()["name"]
