"""Tests for the FastAPI server."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from reroute.api.server import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    return TestClient(app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_info(client):
    r = client.get("/api/info")
    assert r.status_code == 200
    data = r.json()
    assert "model" in data
    assert data["model"]["auc"] > 0.85


def test_generate_endpoint(client):
    r = client.post("/api/generate", json={
        "n_passengers": 15,
        "n_recovery_flights": 3,
        "delay_min": 90,
        "seed": 42,
    })
    assert r.status_code == 200
    data = r.json()
    assert "scenario_id" in data
    assert "passengers" in data
    assert len(data["passengers"]) == 15
    assert "results" in data
    assert "manual" in data["results"]
    assert "lp" in data["results"]


def test_solve_with_custom_coefficients(client):
    # First generate
    gen = client.post("/api/generate", json={
        "n_passengers": 10, "n_recovery_flights": 3,
        "delay_min": 100, "seed": 7,
    }).json()
    # Then re-solve with custom coefficients
    r = client.post("/api/solve", json={
        "scenario": gen["scenario_full"],
        "coefficients": {
            "alpha_yield": 1.0,
            "beta_spill": 2.0,  # Aggressively avoid downgrades
            "delta_harm": 1.5,
            "lambda_miss": 1.0,
            "miss_fixed_cost_usd": 250.0,
        }
    })
    assert r.status_code == 200
    data = r.json()
    assert "results" in data


def test_generate_validation(client):
    """Out-of-range passenger count should 422."""
    r = client.post("/api/generate", json={"n_passengers": 200})
    assert r.status_code == 422
