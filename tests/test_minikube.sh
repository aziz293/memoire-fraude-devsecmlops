# tests/test_minikube.sh — Exécuter : bash tests/test_minikube.sh
#!/bin/bash
echo "=================================================="
echo "TEST 4 — MINIKUBE KUBERNETES"
echo "=================================================="

# 1. État du cluster
echo "\n[1/6] État du cluster Minikube"
minikube status
kubectl cluster-info

# 2. Pods en cours d'exécution
echo "\n[2/6] Pods namespace mlops"
kubectl get pods -n mlops -o wide

# 3. Services exposés
echo "\n[3/6] Services"
kubectl get services -n mlops

# 4. Test du scaling horizontal
echo "\n[4/6] Test scaling : 2 → 3 réplicas"
kubectl scale deployment fraud-detector --replicas=3 -n mlops
sleep 15
kubectl get pods -n mlops

# 5. Test health check via Minikube
echo "\n[5/6] Health check API via Minikube"
SERVICE_URL=$(minikube service fraud-detector-service -n mlops --url)
curl -s $SERVICE_URL/health | python3 -m json.tool

# 6. Ressources consommées
echo "\n[6/6] Consommation ressources"
kubectl top pods -n mlops 2>/dev/null || echo "metrics-server non disponible"
kubectl top nodes

