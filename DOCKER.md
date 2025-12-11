# Multi-Disease Detection - Docker Setup

## Quick Start
```bash
docker-compose up --build
```

## Services
- **FastAPI Backend**: http://localhost:8001
- **Streamlit Frontend**: http://localhost:8501

## Individual Build
```bash
# Build backend only
docker build -f Dockerfile.backend -t mdd-backend .

# Build frontend only  
docker build -f Dockerfile.frontend -t mdd-frontend .
```

## GPU Support
For GPU support, modify docker-compose.yml to use nvidia-docker runtime.
