# Stage 1 : builder
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
# On installe dans un dossier neutre /install
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# Stage 2 : production
FROM python:3.11-slim AS production

# Création de l'utilisateur
RUN groupadd -r mlservice && useradd -r -g mlservice -u 1000 mlservice

WORKDIR /app

# Copier les dépendances depuis /install vers /usr/local (accessible à tous)
COPY --from=builder /install /usr/local

# Copier le code source et les artefacts
COPY src/ ./src/
COPY artifacts/ ./artifacts/

# Permissions
RUN chown -R mlservice:mlservice /app
USER mlservice

EXPOSE 8080

# Utilisation de 127.0.0.1 pour le Healthcheck interne
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://127.0.0.1:8080/health || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
