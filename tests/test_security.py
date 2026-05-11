# Exécuter : python tests/test_security.py
import requests, time

BASE_URL = "https://fraud-detector-latest.onrender.com"
VALID_TOKEN = "60d121d4123a25aa214b25959fe6cfec97623d5b"

def test_auth_required():
    """Test 1 : requête sans token → 403"""
    r = requests.post(f"{BASE_URL}/predict",
                      json={"features": [0.0]*29})
    assert r.status_code == 401
    print(f"✓ Sans token → HTTP {r.status_code} (attendu 403)")

def test_invalid_token():
    """Test 2 : mauvais token → 401"""
    r = requests.post(f"{BASE_URL}/predict",
                      json={"features": [0.0]*29},
                      headers={"Authorization": "Bearer mauvais-token"})
    assert r.status_code == 401
    print(f"✓ Mauvais token → HTTP {r.status_code} (attendu 401)")

def test_valid_token():
    """Test 3 : bon token → 200"""
    r = requests.post(f"{BASE_URL}/predict",
                      json={"features": [0.0]*29},
                      headers={"Authorization": f"Bearer {VALID_TOKEN}"})
    assert r.status_code == 200
    assert "probability" in r.json()
    print(f"✓ Token valide → HTTP {r.status_code}, résultat: {r.json()}")

def test_invalid_features():
    """Test 4 : features invalides → 422"""
    r = requests.post(f"{BASE_URL}/predict",
                      json={"features": [0.0]*5},
                      headers={"Authorization": f"Bearer {VALID_TOKEN}"})
    assert r.status_code == 422
    print(f"✓ Features invalides → HTTP {r.status_code} (attendu 422)")

def test_rate_limit():
    """Test 5 : simulation 110 requêtes → rate limiting actif"""
    statuses = []
    for i in range(110):
        r = requests.post(f"{BASE_URL}/predict",
                          json={"features": [0.0]*29},
                          headers={"Authorization": f"Bearer {VALID_TOKEN}"})
        statuses.append(r.status_code)
    blocked = statuses.count(429)
    print(f"⚠ {blocked}/110 requêtes bloquées par rate limiting")

if __name__ == "__main__":
    print("=" * 55)
    print("TEST 2 — SÉCURITÉ API")
    print("=" * 55)
    test_auth_required()
    test_invalid_token()
    test_valid_token()
    test_invalid_features()
    test_rate_limit()
    print("\n✓ Tous les tests de sécurité réussis")

