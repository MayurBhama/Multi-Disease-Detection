# src/api/router_predict.py
"""
Prediction Router
=================
POST /predict endpoint for image classification.
"""

import os
import tempfile
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException

from src.meta_classifier import MetaClassifier
from src.api.schemas import PredictionResponse, ErrorResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Prediction"])

# Global classifier instance (lazy loaded)
_classifier: Optional[MetaClassifier] = None


def get_classifier() -> MetaClassifier:
    """Get or create MetaClassifier instance."""
    global _classifier
    if _classifier is None:
        logger.info("Initializing MetaClassifier...")
        _classifier = MetaClassifier()
    return _classifier


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Predict disease from image",
    description="Upload an image and get disease prediction with confidence scores."
)
async def predict(
    file: UploadFile = File(..., description="Image file to analyze"),
    disease_type: str = Form(..., description="Disease type: brain_mri, pneumonia, or retina"),
    generate_gradcam: bool = Form(False, description="Generate Grad-CAM overlay")
):
    """
    Predict disease from uploaded image.
    
    - **file**: Image file (PNG, JPG, JPEG)
    - **disease_type**: One of `brain_mri`, `pneumonia`, `retina`
    - **generate_gradcam**: If true, generate Grad-CAM and return URL
    """
    # Validate disease type
    valid_types = ["brain_mri", "pneumonia", "retina"]
    if disease_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid disease_type. Must be one of: {valid_types}"
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (PNG, JPG, JPEG)"
        )
    
    try:
        # Save uploaded file temporarily
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            classifier = get_classifier()
            
            if generate_gradcam:
                # Use explain() which includes prediction + Grad-CAM
                result = classifier.explain(tmp_path, disease_type)
                
                gradcam_url = None
                if "gradcam" in result and "overlay_path" in result["gradcam"]:
                    # Convert local path to URL
                    overlay_path = result["gradcam"]["overlay_path"]
                    gradcam_url = f"/static/{overlay_path.replace(os.sep, '/')}"
                
                return PredictionResponse(
                    disease_type=disease_type,
                    predicted_class=result["prediction"]["class"],
                    class_id=result["prediction"]["class_id"],
                    confidence=result["prediction"]["confidence"],
                    probabilities=result["prediction"]["probabilities"],
                    gradcam_url=gradcam_url
                )
            else:
                # Just prediction
                result = classifier.predict(tmp_path, disease_type)
                
                return PredictionResponse(
                    disease_type=result["disease_type"],
                    predicted_class=result["predicted_class"],
                    class_id=result["class_id"],
                    confidence=result["confidence"],
                    probabilities=result["probabilities"],
                    gradcam_url=None
                )
                
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
