# ==============================================================================
#  NYC TAXI MLOPS PROJECT - MASTER MAKEFILE
# ==============================================================================

PYTHON = python
PIP = pip
DOCKER_COMPOSE = docker-compose
KUBECTL = kubectl
MINIKUBE = minikube
K8S_DIR = k8s
VENV = venv
RM = rmdir /s /q

.PHONY: help install ingest train test clean docker-up docker-down k8s-start k8s-build k8s-up start-all-docker start-all-k8s

# ==============================================================================
#  COMMANDS
# ==============================================================================

help:
	@echo ---------------------------------------------------
	@echo  NYC TAXI MLOPS PROJECT - COMMAND CENTER
	@echo ---------------------------------------------------
	@echo  [ ONE-CLICK ACTIONS ]
	@echo  make start-all-docker : Full Setup (Ingest + Build + Run Docker)
	@echo  make start-all-k8s    : Full Setup (Ingest + Minikube + Build + Deploy)
	@echo ---------------------------------------------------
	@echo  [ SETUP ]
	@echo  make install          : Install dependencies
	@echo  make ingest           : Download data from Drive (Zero-Touch)
	@echo  make clean-venv       : Remove virtual environment
	@echo ---------------------------------------------------
	@echo  [ MODEL / TESTS ]
	@echo  make train            : Check data .. Train model locally
	@echo  make test             : Run unit tests
	@echo ---------------------------------------------------
	@echo  [ DOCKER COMPOSE ]
	@echo  make docker-build     : Build Docker images
	@echo  make docker-up        : Check data .. Start API and UI
	@echo  make docker-down      : Stop containers
	@echo  make docker-logs      : Show Logs
	@echo ---------------------------------------------------
	@echo  [ KUBERNETES / MINIKUBE ]
	@echo  make k8s-start        : Start Minikube
	@echo  make k8s-build        : Build images INSIDE Minikube
	@echo  make k8s-up           : Deploy to K8s
	@echo  make k8s-down         : Stop K8s
	@echo  make k8s-forward      : Port-Forward UI (8501)
	@echo ---------------------------------------------------
	@echo  [ UTILS ]
	@echo  make clean            : Clean Docker system
	@echo  make format           : Format code (Black/Isort)
	@echo ---------------------------------------------------

# ==============================================================================
#  SETUP & DATA INGESTION
# ==============================================================================

install:
	@echo "Installing Dependencies..."
	$(PIP) install -r requirements/base.txt
	@echo "Installation Complete."

ingest:
	@echo "------------------------------------------------"
	@echo "CHECKING DATA INTEGRITY..."
	@echo "------------------------------------------------"
	$(PYTHON) -m src.components.data_ingestion

clean-venv:
	@echo "Removing virtual environment..."
	$(RM) $(VENV)

# ==============================================================================
#  ONE-CLICK WORKFLOWS
# ==============================================================================

# 1. DOCKER
start-all-docker: ingest docker-down
	@echo "------------------------------------------------"
	@echo "STARTING FULL DOCKER PIPELINE..."
	@echo "------------------------------------------------"
	$(DOCKER_COMPOSE) up --build -d
	@echo "UI is ready at http://localhost:8501"

# 2. KUBERNETES
start-all-k8s: ingest k8s-start k8s-build k8s-up
	@echo "------------------------------------------------"
	@echo "FULL KUBERNETES PIPELINE DEPLOYED!"
	@echo "------------------------------------------------"
	@echo "Waiting for pods to be ready..."
	@echo "Tip: Use 'make k8s-forward' to access the UI if needed."

# ==============================================================================
#  MODEL & TESTS
# ==============================================================================

train: ingest
	@echo "STARTING LOCAL TRAINING..."
	$(PYTHON) -m src.pipelines.training_pipeline

test:
	@echo "Running Tests..."
	pytest

# ==============================================================================
#  DOCKER
# ==============================================================================

docker-build:
	$(DOCKER_COMPOSE) build

docker-up: ingest
	@echo "Starting Services..."
	$(DOCKER_COMPOSE) up --build -d
	@echo "UI is ready at http://localhost:8501"

docker-down:
	$(DOCKER_COMPOSE) down

docker-logs:
	$(DOCKER_COMPOSE) logs -f

# ==============================================================================
#  KUBERNETES
# ==============================================================================

k8s-start:
	$(MINIKUBE) start --driver=docker --memory 6144 --cpus 4
	$(MINIKUBE) addons enable metrics-server

k8s-build:
	@echo "Switching to Minikube Docker Env..."
	powershell -Command "& minikube -p minikube docker-env --shell powershell | Invoke-Expression; docker-compose build"

k8s-up:
	@echo "Deploying to Kubernetes..."
	$(KUBECTL) apply -f $(K8S_DIR)/

k8s-down:
	$(KUBECTL) delete -f $(K8S_DIR)/

k8s-restart:
	$(KUBECTL) rollout restart deployment/api-deployment
	$(KUBECTL) rollout restart deployment/ui-deployment

k8s-forward:
	@echo "Forwarding Port 8501..."
	$(KUBECTL) port-forward service/ui-service 8501:8501

# ==============================================================================
#  CLEANING & FORMAT
# ==============================================================================

clean:
	docker system prune -f

format:
	@echo "Formatting code..."
	$(PYTHON) -m isort .
	$(PYTHON) -m black .