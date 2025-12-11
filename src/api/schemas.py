# src/api/schemas.py
"""
Pydantic Response Models for FastAPI
====================================
"""

from typing import Dict, Optional, List
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "1.0.0"
    models_loaded: List[str] = []


class PredictionResponse(BaseModel):
    """Prediction response schema."""
    disease_type: str = Field(..., description="Type of disease analyzed")
    predicted_class: str = Field(..., description="Predicted class name")
    class_id: int = Field(..., description="Predicted class index")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    probabilities: Dict[str, float] = Field(..., description="All class probabilities")
    gradcam_url: Optional[str] = Field(None, description="Grad-CAM image URL")
    
    class Config:
        json_schema_extra = {
            "example": {
                "disease_type": "brain_mri",
                "predicted_class": "glioma",
                "class_id": 0,
                "confidence": 0.95,
                "probabilities": {"glioma": 0.95, "meningioma": 0.03},
                "gradcam_url": "/static/gradcam/brain_mri/image_overlay.png"
            }
        }


class GradCAMResponse(BaseModel):
    """Grad-CAM response schema."""
    disease_type: str
    predicted_class: str
    confidence: float
    heatmap_url: str
    overlay_url: str
    comparison_url: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: Optional[str] = None
