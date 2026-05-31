from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, field_validator
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

# Détection dynamique du nom du service pour différencier Local et Render dans Grafana
SERVICE_NAME = os.getenv("SERVICE_NAME", "fastapi-api-local")

# 🌟 Métriques Prometheus mises à jour avec les vrais labels attendus
# Remplacement de 'endpoint' par 'exported_endpoint' et ajout de 'service' et 'client_ip'
REQUESTS = Counter("api_requests_total", "Total requests", ["exported_endpoint", "status", "service", "client_ip"])
LATENCY  = Histogram("api_latency_seconds", "Request latency", ["exported_endpoint", "service"])
PREDICTIONS = Counter("predictions_total", "Predictions", ["result", "service"])

# Token d'authentification
API_TOKEN = os.getenv("API_TOKEN", "60d121d4123a25aa214b25959fe6cfec97623d5b")

def verify_token(creds: HTTPAuthorizationCredentials = Depends(security)):
    if creds.credentials != API_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid authentication token")
    return creds.credentials

class TransactionFeatures(BaseModel):
    features: list[float]

    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        if len(v) != 29:   # 30 features - Time = 29
            raise ValueError(f"Expected 29 features, got {len(v)}")
        return v

# 🌟 Middleware Global pour intercepter l'IP sur Render et suivre le trafic (Même les 404/405/401)
@app.middleware("http")
async def monitor_render_security(request: Request, call_next):
    start_time = time.time()
    
    # Extraction stricte de l'IP sur Render uniquement
    x_forwarded_for = request.headers.get("x-forwarded-for")
    if x_forwarded_for:
        client_ip = x_forwarded_for.split(",")[0].strip()
    else:
        client_ip = "local-environment"

    endpoint = request.url.path

    # Ignorer la route /metrics pour ne pas fausser les stats de sécurité
    if endpoint == "/metrics":
        return await call_next(request)

    try:
        response = await call_next(request)
        status_code = str(response.status_code)
        
        # Enregistrement automatique du trafic global
        REQUESTS.labels(exported_endpoint=endpoint, status=status_code, service=SERVICE_NAME, client_ip=client_ip).inc()
        LATENCY.labels(exported_endpoint=endpoint, service=SERVICE_NAME).observe(time.time() - start_time)
        
        return response
    except Exception as e:
        # En cas de crash serveur non géré
        REQUESTS.labels(exported_endpoint=endpoint, status="500", service=SERVICE_NAME, client_ip=client_ip).inc()
        LATENCY.labels(exported_endpoint=endpoint, service=SERVICE_NAME).observe(time.time() - start_time)
        raise e

# 🌟 Gestionnaire d'exceptions pour capter les erreurs d'authentification ou de validation
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    x_forwarded_for = request.headers.get("x-forwarded-for")
    client_ip = x_forwarded_for.split(",")[0].strip() if x_forwarded_for else "local-environment"
    
    # Incrémente les 401, 404, 405 ici aussi
    REQUESTS.labels(exported_endpoint=request.url.path, status=str(exc.status_code), service=SERVICE_NAME, client_ip=client_ip).inc()
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.post("/predict")
async def predict(data: TransactionFeatures, token=Depends(verify_token)):
    try:
        X = np.array(data.features).reshape(1, -1)
        proba = float(model.predict_proba(X)[0][1])
        decision = "fraud" if proba > 0.5 else "legitimate"
        
        # Métrique spécifique au modèle
        PREDICTIONS.labels(result=decision, service=SERVICE_NAME).inc()
        
        return {"probability": round(proba, 3), "decision": decision, "threshold": 0.5}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "RandomForestClassifier", "version": "1.0.0"}

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
