# src/api/main.py
"""
FastAPI Application for Multi-Disease Detection
================================================
Production-ready API with prediction and Grad-CAM endpoints.

Run with:
    uvicorn src.api.main:app --reload --port 8000
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.schemas import HealthResponse
from src.api.router_predict import router as predict_router
from src.api.router_gradcam import router as gradcam_router
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info("=" * 50)
    logger.info("Multi-Disease Detection API Starting...")
    logger.info("=" * 50)
    
    # Create output directories
    os.makedirs("outputs/gradcam", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    yield
    
    logger.info("API Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Multi-Disease Detection API",
    description="""
    ## Production API for Medical Image Classification
    
    Supports three disease types:
    - **Brain MRI**: Tumor classification (glioma, meningioma, notumor, pituitary)
    - **Pneumonia**: Chest X-ray classification (NORMAL, PNEUMONIA)
    - **Retina**: Diabetic retinopathy severity (No DR, Mild, Moderate, Severe, Proliferative)
    
    ### Features
    - Image prediction with confidence scores
    - Grad-CAM explainability visualizations
    - EfficientNet ensemble for retina analysis
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount static files for Grad-CAM images
# Creates symlink from /static to outputs/
app.mount("/static", StaticFiles(directory="outputs"), name="static")


# Include routers
app.include_router(predict_router)
app.include_router(gradcam_router)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint"
)
async def health():
    """Check API health and list loaded models."""
    from src.api.router_predict import _classifier
    
    models_loaded = []
    if _classifier is not None:
        models_loaded = _classifier.get_supported_diseases()
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=models_loaded
    )


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API info."""
    return {
        "message": "Multi-Disease Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "gradcam": "/gradcam"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
