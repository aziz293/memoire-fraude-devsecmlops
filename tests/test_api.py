import os
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

# 1. Configurer l'environnement AVANT l'import de l'app
os.environ["API_TOKEN"] = "test-token"
from src.api import app

client = TestClient(app)
HEADERS = {"Authorization": "Bearer test-token"}

# --- Tests de l'API ---

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
    payload = {"features": [0.0] * 10}
    r = client.post("/predict", json=payload, headers=HEADERS)
    assert r.status_code == 422

def test_predict_no_token():
    payload = {"features": [0.0] * 29}
    r = client.post("/predict", json=payload)
    assert r.status_code == 401

def test_predict_malformed_data():
    payload = {"features": ["invalid"] * 29}
    r = client.post("/predict", json=payload, headers=HEADERS)
    assert r.status_code == 422

def test_predict_empty_payload():
    r = client.post("/predict", json={}, headers=HEADERS)
    assert r.status_code == 422

# --- Test de la logique de Drift ---

def test_drift_detection_logic():
    from src.drift_detection import run_drift_report
    
    # Création de colonnes nommées pour correspondre à ce qu'Evidently attend
    cols = [f"col_{i}" for i in range(29)]
    df_ref = pd.DataFrame(np.random.randn(100, 29), columns=cols)
    df_curr = pd.DataFrame(np.random.randn(100, 29) + 5, columns=cols)
    
    # On crée un CSV temporaire car ta fonction run_drift_report lit un chemin (path)
    ref_path = "tests/temp_reference.csv"
    df_ref.to_csv(ref_path, index=False)
    
    # Appel de la fonction avec les bons types
    drift_detected, drift_share = run_drift_report(ref_path, df_curr)
    
    assert isinstance(drift_detected, bool)
    assert isinstance(drift_share, float)
    
    # Nettoyage
    if os.path.exists(ref_path):
        os.remove(ref_path)
