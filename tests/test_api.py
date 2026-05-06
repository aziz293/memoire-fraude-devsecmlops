import os

os.environ["API_TOKEN"] = "test-token"

from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)
HEADERS = {"Authorization": "Bearer test-token"}

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_predict_valid():
    payload = {"features": [0.0] * 29}
    r = client.post("/predict", json=payload, headers=HEADERS)
    assert r.status_code == 200
    assert "probability" in r.json()
    assert r.json()["decision"] in ["fraud", "legitimate"]

def test_predict_invalid_features():
    payload = {"features": [0.0] * 10}  # Mauvais nombre de features
    r = client.post("/predict", json=payload, headers=HEADERS)
    assert r.status_code == 422

def test_predict_no_token():
    payload = {"features": [0.0] * 29}
    r = client.post("/predict", json=payload)  # Sans token
    assert r.status_code == 403

