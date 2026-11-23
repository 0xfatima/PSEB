"""
Tests for the API, using FastAPI's TestClient (no server needed).

Run after training:
    python train.py
    pytest -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from sklearn.datasets import load_digits

from api import app

client = TestClient(app)
digits = load_digits()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_real_image():
    r = client.post("/predict", json={"pixels": digits.data[0].tolist()})
    assert r.status_code == 200
    body = r.json()
    assert body["digit"] == 0
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["probabilities"]) == 10


def test_wrong_number_of_pixels():
    r = client.post("/predict", json={"pixels": [0.0] * 10})
    assert r.status_code == 422
    assert "64" in r.json()["error"]


def test_pixel_out_of_range():
    r = client.post("/predict", json={"pixels": [100.0] * 64})
    assert r.status_code == 422
    assert "between" in r.json()["error"]


def test_not_numbers():
    r = client.post("/predict", json={"pixels": ["a"] * 64})
    assert r.status_code == 422          # rejected by pydantic, no crash
