# Exécuter : python tests/test_model_performance.py
import joblib, numpy as np, pandas as pd, json, time
from sklearn.metrics import (
    classification_report, f1_score, precision_score,
    recall_score, roc_auc_score, accuracy_score,
    confusion_matrix, average_precision_score
)
from sklearn.model_selection import train_test_split

# ── Chargement ───────────────────────────────────────────
print("=" * 60)
print("TEST 1 — PERFORMANCE DU MODÈLE")
print("=" * 60)

model = joblib.load("artifacts/fraud_model.pkl")
df = pd.read_csv("artifacts/reference_data.csv")

# Reconstituer X_test / y_test de manière reproductible
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
full = pd.read_csv(url)
X = full.drop(["Class","Time"], axis=1)
y = full["Class"]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2,
                                         random_state=42, stratify=y)

# ── Prédictions ──────────────────────────────────────────
start = time.time()
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
latency_total = (time.time() - start) * 1000
latency_per   = latency_total / len(X_test)

# ── Métriques ────────────────────────────────────────────
metrics = {
    "accuracy"        : accuracy_score(y_test, y_pred),
    "precision"       : precision_score(y_test, y_pred),
    "recall"          : recall_score(y_test, y_pred),
    "f1_score"        : f1_score(y_test, y_pred),
    "auc_roc"         : roc_auc_score(y_test, y_proba),
    "avg_precision"   : average_precision_score(y_test, y_proba),
    "latency_ms_total": round(latency_total, 2),
    "latency_ms_per"  : round(latency_per, 4),
}

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# ── Affichage ────────────────────────────────────────────
print(f"\n{'Métrique':<25} {'Valeur':>10}")
print("-" * 37)
for k, v in metrics.items():
    bar = "✓" if v > 0.85 else "⚠"
    print(f"{k:<25} {v:>10.4f}  {bar}")

print(f"\nMatrice de confusion:")
print(f"  TN={tn}  FP={fp}")
print(f"  FN={fn}  TP={tp}")

print("\n" + classification_report(y_test, y_pred,
      target_names=["Légitime","Fraude"]))

# ── Sauvegarde JSON ──────────────────────────────────────
with open("results/model_performance.json", "w") as f:
    json.dump({**metrics, "confusion_matrix": cm.tolist()}, f, indent=2)
print("\n✓ Résultats sauvegardés dans results/model_performance.json")

