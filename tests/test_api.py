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
    assert r.status_code == 401


def test_predict_malformed_data():
    # Test avec des chaînes de caractères au lieu de nombres
    payload = {"features": ["invalid"] * 29}
    r = client.post("/predict", json=payload, headers=HEADERS)
    assert r.status_code == 422 # Erreur de validation Pydantic

def test_drift_detection_logic():
    # On importe la logique de drift ici pour augmenter le coverage de ce fichier
    from src.drift_detection import detect_drift
    import pandas as pd
    import numpy as np
    
    # Création de données fictives (référence vs actuel)
    ref = pd.DataFrame(np.random.randn(100, 29))
    curr = pd.DataFrame(np.random.randn(100, 29) + 5) # On ajoute un décalage (drift)
    
    drift_report = detect_drift(ref, curr)
    assert "drift_detected" in drift_report
    assert isinstance(drift_report["drift_detected"], bool)
