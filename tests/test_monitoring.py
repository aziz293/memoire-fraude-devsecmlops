import requests, time, random, numpy as np, pandas as pd
from src.drift_detection import run_drift_report

BASE_URL = "https://fraud-detector-latest.onrender.com"  # ou URL Render
TOKEN = "60d121d4123a25aa214b25959fe6cfec97623d5b"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# ── Test 1 : Génération de trafic pour Prometheus ────────
print("[1/4] Génération de 200 requêtes de test...")
results = {"fraud": 0, "legitimate": 0, "error": 0}
latencies = []

for i in range(200):
    # Features aléatoires simulant des transactions
    features = [round(random.gauss(0, 1), 4) for _ in range(28)]
    features.append(round(random.uniform(1, 5000), 2))  # Amount
    start = time.time()
    r = requests.post(f"{BASE_URL}/predict",
                      json={"features": features},
                      headers=HEADERS)
    latencies.append((time.time() - start) * 1000)
    if r.status_code == 200:
        results[r.json()["decision"]] += 1
    else:
        results["error"] += 1

p95 = np.percentile(latencies, 95)
p99 = np.percentile(latencies, 99)
print(f"  Légitimes    : {results['legitimate']}")
print(f"  Fraudes      : {results['fraud']}")
print(f"  Erreurs      : {results['error']}")
print(f"  Latence P50  : {np.percentile(latencies, 50):.1f} ms")
print(f"  Latence P95  : {p95:.1f} ms")
print(f"  Latence P99  : {p99:.1f} ms")

# ── Test 2 : Métriques Prometheus disponibles ────────────
print("\n[2/4] Vérification métriques Prometheus...")
r = requests.get(f"{BASE_URL}/metrics")
metrics_raw = r.text
expected = ["api_requests_total","api_latency_seconds","predictions_total"]
for m in expected:
    found = m in metrics_raw
    status = "✓ trouvée" if found else "✗ manquante"
    print(f"  {m}: {status}")

# ── Test 3 : Détection de dérive normale ─────────────────
print("\n[3/4] Test dérive — données normales (pas de dérive attendue)...")
ref = pd.read_csv("artifacts/reference_data.csv")
# Légère variation aléatoire (normale)
current_normal = ref.copy() + np.random.normal(0, 0.1, ref.shape)
drift, share = run_drift_report(
    "artifacts/reference_data.csv",
    current_normal,
    "results/drift_normal.html"
)
print(f"  Dérive détectée : {drift} | Features driftées : {share*100:.1f}%")

# ── Test 4 : Détection de dérive simulée ─────────────────
print("\n[4/4] Test dérive — simulation dérive significative...")
current_drifted = ref.copy()
# Simulation drift réaliste : changement distribution + bruit
current_drifted = current_drifted * 1.5 + np.random.normal(3, 1.5, ref.shape)
drift2, share2 = run_drift_report(
    "artifacts/reference_data.csv",
    current_drifted,
    "results/drift_simulated.html"
)
print(f"  Dérive détectée : {drift2} | Features driftées : {share2*100:.1f}%")
if drift2: print("  ⚠ ALERTE : réentraînement recommandé")

