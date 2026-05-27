from fastapi import FastAPI, HTTPException, Depends, status, Request  # 👈 Corrigé : Import de Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, field_validator  # 👈 Corrigé : field_validator pour Pydantic V2
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
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

# Détection dynamique du nom du service pour différencier Local et Render dans Grafana
SERVICE_NAME = os.getenv("SERVICE_NAME", "fastapi-api-local")

# Token d'authentification
API_TOKEN = os.getenv("API_TOKEN", "60d121d4123a25aa214b25959fe6cfec97623d5b")

def verify_token(creds: HTTPAuthorizationCredentials = Depends(security)):
    if creds.credentials != API_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid authentication token")
    return creds.credentials

class TransactionFeatures(BaseModel):
    features: list[float]

    @field_validator("features")  # 👈 Corrigé : Syntaxe moderne Pydantic V2
    @classmethod
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


# --- INTERCEPTION DES ERREURS POUR PROMETHEUS ---

# 1. Erreurs d'authentification ou HTTP classiques (ex: 401, 403, 404)
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    REQUESTS.labels(
        endpoint=request.url.path,
        status=str(exc.status_code)
    ).inc()
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

# 2. Erreurs de validation de payload / types de données manquants (Erreur 422)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    REQUESTS.labels(
        endpoint=request.url.path,
        status=str(status.HTTP_422_UNPROCESSABLE_ENTITY)
    ).inc()
    
    # 🌟 CORRECTION ROBUSTE : On extrait proprement les erreurs au format dictionnaire
    # Si Pydantic bloque sur un objet non-sérialisable, on extrait son message brut.
    cleaned_errors = []
    for err in exc.errors():
        cleaned_err = err.copy()
        if "ctx" in cleaned_err and "error" in cleaned_err["ctx"]:
            # On convertit l'objet ValueError/Exception interne en chaîne de caractères
            cleaned_err["ctx"]["error"] = str(cleaned_err["ctx"]["error"])
        cleaned_errors.append(cleaned_err)
        
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": cleaned_errors},
    )
