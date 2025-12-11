# src/api/router_gradcam.py
"""
Grad-CAM Router
===============
POST /gradcam endpoint for visual explanations.
"""

import os
import tempfile
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException

from src.meta_classifier import MetaClassifier
from src.api.schemas import GradCAMResponse, ErrorResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Explainability"])

# Reuse classifier from router_predict
from src.api.router_predict import get_classifier


@router.post(
    "/gradcam",
    response_model=GradCAMResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Generate Grad-CAM explanation",
    description="Upload an image and get Grad-CAM visualization showing model attention."
)
async def gradcam(
    file: UploadFile = File(..., description="Image file to analyze"),
    disease_type: str = Form(..., description="Disease type: brain_mri, pneumonia, or retina")
):
    """
    Generate Grad-CAM visualization for uploaded image.
    
    Returns URLs to:
    - Heatmap image
    - Overlay image (heatmap on original)
    - Comparison image (original | heatmap | overlay)
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
            
            # Generate Grad-CAM
            result = classifier.explain(tmp_path, disease_type)
            
            # Convert paths to URLs
            def path_to_url(path: str) -> str:
                # outputs/gradcam/... -> /static/gradcam/...
                rel_path = path.replace("outputs/", "").replace(os.sep, "/")
                return f"/static/{rel_path}"
            
            gradcam_info = result.get("gradcam", {})
            
            heatmap_url = path_to_url(gradcam_info.get("heatmap_path", ""))
            overlay_url = path_to_url(gradcam_info.get("overlay_path", ""))
            comparison_url = None
            if "comparison_path" in gradcam_info:
                comparison_url = path_to_url(gradcam_info["comparison_path"])
            
            return GradCAMResponse(
                disease_type=disease_type,
                predicted_class=result["prediction"]["class"],
                confidence=result["prediction"]["confidence"],
                heatmap_url=heatmap_url,
                overlay_url=overlay_url,
                comparison_url=comparison_url
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
        logger.error(f"Grad-CAM generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Grad-CAM failed: {str(e)}")
