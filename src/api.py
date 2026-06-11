from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, field_validator
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import joblib, numpy as np, os, time, logging, random

app = FastAPI(title="Fraud Detection API", version="1.0.0",
              description="DevSecMLOps — Random Forest Classifier")
security = HTTPBearer()

model = joblib.load("artifacts/fraud_model.pkl")

# ── Chargement des stats d'entraînement ──────────────────────────────────────
try:
    _stats = joblib.load("artifacts/feature_stats.pkl")
    TRAIN_MEAN = _stats["mean"]
    TRAIN_STD  = _stats["std"]
    # 🌟 CORRECTION 3 : Si un écart-type est nul, on le remplace par 1 pour éviter l'explosion du score Z
    TRAIN_STD  = np.where(TRAIN_STD == 0, 1.0, TRAIN_STD)
except FileNotFoundError:
    TRAIN_MEAN = None
    TRAIN_STD  = None
    logging.warning("[SECURITY] feature_stats.pkl not found — input validation disabled")

SERVICE_NAME = os.getenv("SERVICE_NAME", "fastapi-api-local")
API_TOKEN    = os.getenv("API_TOKEN", "60d121d4123a25aa214b25959fe6cfec97623d5b")

REQUESTS    = Counter("api_requests_total", "Total requests",
                      ["exported_endpoint", "status", "service", "client_ip"])
LATENCY     = Histogram("api_latency_seconds", "Request latency",
                        ["exported_endpoint", "service"])
PREDICTIONS = Counter("predictions_total", "Predictions", ["result", "service"])

SUSPICIOUS  = Counter("suspicious_requests_total",
                      "Out-of-distribution requests detected",
                      ["service", "client_ip"])

def verify_token(creds: HTTPAuthorizationCredentials = Depends(security)):
    if creds.credentials != API_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid authentication token")
    return creds.credentials

def is_suspicious(features: list[float], threshold: float = 5.0) -> bool:
    """Retourne True si au moins une feature dépasse `threshold` écarts-types."""
    if TRAIN_MEAN is None:
        return False
    x = np.array(features)
    # 🌟 CORRECTION 3 (suite) : Plus besoin de ajouter 1e-8 risqué, TRAIN_STD est nettoyé
    z_scores = np.abs((x - TRAIN_MEAN) / TRAIN_STD)
    return bool(np.any(z_scores > threshold))

class TransactionFeatures(BaseModel):
    features: list[float]

    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        if len(v) != 29:
            raise ValueError(f"Expected 29 features, got {len(v)}")
        return v

@app.middleware("http")
async def monitor_render_security(request: Request, call_next):
    start_time = time.time()
    x_forwarded_for = request.headers.get("x-forwarded-for")
    client_ip = x_forwarded_for.split(",")[0].strip() if x_forwarded_for else "local-environment"
    endpoint = request.url.path

    if endpoint == "/metrics":
        return await call_next(request)

    try:
        response = await call_next(request)
        status_code = str(response.status_code)
        REQUESTS.labels(exported_endpoint=endpoint, status=status_code,
                        service=SERVICE_NAME, client_ip=client_ip).inc()
        LATENCY.labels(exported_endpoint=endpoint, service=SERVICE_NAME).observe(time.time() - start_time)
        return response
    except Exception as e:
        REQUESTS.labels(exported_endpoint=endpoint, status="500",
                        service=SERVICE_NAME, client_ip=client_ip).inc()
        LATENCY.labels(exported_endpoint=endpoint, service=SERVICE_NAME).observe(time.time() - start_time)
        raise e

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    x_forwarded_for = request.headers.get("x-forwarded-for")
    client_ip = x_forwarded_for.split(",")[0].strip() if x_forwarded_for else "local-environment"
    REQUESTS.labels(exported_endpoint=request.url.path, status=str(exc.status_code),
                    service=SERVICE_NAME, client_ip=client_ip).inc()
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.post("/predict")
async def predict(data: TransactionFeatures, request: Request, token=Depends(verify_token)):
    x_forwarded_for = request.headers.get("x-forwarded-for")
    client_ip = x_forwarded_for.split(",")[0].strip() if x_forwarded_for else "local-environment"

    # ── Validation de distribution (Suspicious / Drift) ──────────────────────
    if is_suspicious(data.features):
        logging.warning(
            f"[SECURITY] Out-of-distribution input from {client_ip} — "
            f"first 5 features: {data.features[:5]}"
        )
        SUSPICIOUS.labels(service=SERVICE_NAME, client_ip=client_ip).inc()
        
        # 🌟 AJOUT : On comptabilise cette anomalie/erreur dans les prédictions
        PREDICTIONS.labels(result="error", service=SERVICE_NAME).inc()

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "probability": round(random.uniform(0.1, 0.9), 3),
                "decision":    "unknown",
                "warning":     "input out of expected distribution"
            }
        )

    # ── Inférence normale ─────────────────────────────────────────────────────
    try:
        X      = np.array(data.features).reshape(1, -1)
        proba  = float(model.predict_proba(X)[0][1])
        decision = "fraud" if proba > 0.5 else "legitimate"
        
        # Enregistrement du succès
        PREDICTIONS.labels(result=decision, service=SERVICE_NAME).inc()
        return {"probability": round(proba, 3), "decision": decision, "threshold": 0.5}
        
    except Exception as e:
        # 🌟 AJOUT : On comptabilise le crash du modèle ici
        PREDICTIONS.labels(result="error", service=SERVICE_NAME).inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "RandomForestClassifier", "version": "1.0.0"}

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
