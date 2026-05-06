from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import joblib, numpy as np, os, time

app = FastAPI(title="Fraud Detection API", version="1.0.0",
              description="DevSecMLOps — Random Forest Classifier")
security = HTTPBearer()

# Charger le modèle au démarrage
model = joblib.load("artifacts/fraud_model.pkl")

# Métriques Prometheus
REQUESTS = Counter("api_requests_total", "Total requests", ["endpoint", "status"])
LATENCY  = Histogram("api_latency_seconds", "Request latency", ["endpoint"])
PREDICTIONS = Counter("predictions_total", "Predictions", ["result"])

# Token d'authentification (stocker dans variable d'environnement)
API_TOKEN = os.getenv("API_TOKEN", "changeme-in-production")

def verify_token(creds: HTTPAuthorizationCredentials = Depends(security)):
    if creds.credentials != API_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid authentication token")
    return creds.credentials

class TransactionFeatures(BaseModel):
    features: list[float]

    @validator("features")
    def validate_features(cls, v):
        if len(v) != 29:   # 30 features - Time = 29
            raise ValueError(f"Expected 29 features, got {len(v)}")
        return v

@app.post("/predict")
async def predict(data: TransactionFeatures, token=Depends(verify_token)):
    start = time.time()
    try:
        X = np.array(data.features).reshape(1, -1)
        proba = float(model.predict_proba(X)[0][1])
        decision = "fraud" if proba > 0.5 else "legitimate"
        PREDICTIONS.labels(result=decision).inc()
        REQUESTS.labels(endpoint="/predict", status="200").inc()
        return {"probability": round(proba, 3), "decision": decision, "threshold": 0.5}
    except Exception as e:
        REQUESTS.labels(endpoint="/predict", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        LATENCY.labels(endpoint="/predict").observe(time.time() - start)

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "RandomForestClassifier", "version": "1.0.0"}

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

