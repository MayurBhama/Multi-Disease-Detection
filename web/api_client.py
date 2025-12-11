# web/api_client.py
"""
FastAPI Client Wrapper
======================
Handles all HTTP requests to the FastAPI backend.
"""

import requests
from typing import Dict, Any, Optional, List, Tuple
from io import BytesIO
import time


class APIClient:
    """Client for Multi-Disease Detection FastAPI backend."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8001"):
        self.base_url = base_url.rstrip("/")
        self.timeout = 120  # 2 minutes for model inference
    
    def health_check(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check API health.
        
        Returns:
            Tuple of (is_healthy, response_data)
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=10
            )
            if response.status_code == 200:
                return True, response.json()
            return False, {"error": f"Status {response.status_code}"}
        except requests.exceptions.ConnectionError:
            return False, {"error": "Cannot connect to API server"}
        except requests.exceptions.Timeout:
            return False, {"error": "API request timed out"}
        except Exception as e:
            return False, {"error": str(e)}
    
    def predict(
        self,
        image_bytes: bytes,
        filename: str,
        disease_type: str,
        generate_gradcam: bool = False
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Send image for prediction.
        
        Args:
            image_bytes: Image file content
            filename: Original filename
            disease_type: One of brain_mri, pneumonia, retina
            generate_gradcam: Whether to generate Grad-CAM
            
        Returns:
            Tuple of (success, response_data)
        """
        try:
            files = {
                "file": (filename, BytesIO(image_bytes), "image/png")
            }
            data = {
                "disease_type": disease_type,
                "generate_gradcam": str(generate_gradcam).lower()
            }
            
            response = requests.post(
                f"{self.base_url}/predict",
                files=files,
                data=data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return True, response.json()
            else:
                error_detail = response.json().get("detail", response.text)
                return False, {"error": error_detail}
                
        except requests.exceptions.ConnectionError:
            return False, {"error": "Cannot connect to API. Is the server running?"}
        except requests.exceptions.Timeout:
            return False, {"error": "Request timed out. The image may be too large."}
        except Exception as e:
            return False, {"error": f"Request failed: {str(e)}"}
    
    def gradcam(
        self,
        image_bytes: bytes,
        filename: str,
        disease_type: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Generate Grad-CAM explanation.
        
        Returns:
            Tuple of (success, response_data with image URLs)
        """
        try:
            files = {
                "file": (filename, BytesIO(image_bytes), "image/png")
            }
            data = {
                "disease_type": disease_type
            }
            
            response = requests.post(
                f"{self.base_url}/gradcam",
                files=files,
                data=data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return True, response.json()
            else:
                error_detail = response.json().get("detail", response.text)
                return False, {"error": error_detail}
                
        except requests.exceptions.ConnectionError:
            return False, {"error": "Cannot connect to API server"}
        except requests.exceptions.Timeout:
            return False, {"error": "Grad-CAM generation timed out"}
        except Exception as e:
            return False, {"error": f"Request failed: {str(e)}"}
    
    def get_gradcam_image(self, url_path: str) -> Optional[bytes]:
        """
        Fetch Grad-CAM image from static URL.
        
        Args:
            url_path: Relative URL path (e.g., /static/gradcam/...)
            
        Returns:
            Image bytes or None if failed
        """
        try:
            full_url = f"{self.base_url}{url_path}"
            response = requests.get(full_url, timeout=30)
            if response.status_code == 200:
                return response.content
            return None
        except Exception:
            return None
    
    def predict_batch(
        self,
        images: List[Tuple[bytes, str]],
        disease_type: str,
        progress_callback=None
    ) -> List[Dict[str, Any]]:
        """
        Predict multiple images.
        
        Args:
            images: List of (image_bytes, filename) tuples
            disease_type: Disease type for all images
            progress_callback: Optional callback(current, total)
            
        Returns:
            List of prediction results
        """
        results = []
        total = len(images)
        
        for i, (img_bytes, filename) in enumerate(images):
            if progress_callback:
                progress_callback(i + 1, total)
            
            success, result = self.predict(
                img_bytes, filename, disease_type, generate_gradcam=False
            )
            
            result["filename"] = filename
            result["success"] = success
            results.append(result)
            
            # Small delay to avoid overwhelming the server
            time.sleep(0.1)
        
        return results


# Utility functions
def format_confidence(confidence: float) -> str:
    """Format confidence as percentage string."""
    return f"{confidence * 100:.1f}%"


def validate_image(file) -> Tuple[bool, str]:
    """
    Validate uploaded image file.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if file is None:
        return False, "No file uploaded"
    
    # Check file extension
    valid_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".gif"]
    filename = file.name.lower()
    
    if not any(filename.endswith(ext) for ext in valid_extensions):
        return False, f"Invalid file type. Supported: {', '.join(valid_extensions)}"
    
    # Check file size (max 10MB)
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Reset to beginning
    
    if size > 10 * 1024 * 1024:
        return False, "File too large. Maximum size is 10MB."
    
    if size == 0:
        return False, "File is empty."
    
    return True, ""


def get_disease_info(disease_type: str) -> Dict[str, Any]:
    """Get comprehensive medical information about disease type."""
    info = {
        "brain_mri": {
            "name": "Brain MRI Tumor Classification",
            "description": "Classifies brain MRI scans into tumor types using deep learning",
            "classes": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
            "color": "#6366f1",
            "medical_details": {
                "glioma": {
                    "description": "Gliomas are tumors that originate from glial cells in the brain or spine. They are the most common type of primary brain tumor.",
                    "severity": "High",
                    "recommendation": "Immediate consultation with a neuro-oncologist is recommended. Treatment may include surgery, radiation therapy, and/or chemotherapy.",
                    "prevalence": "Represents about 30% of all brain tumors"
                },
                "meningioma": {
                    "description": "Meningiomas arise from the meninges, the membranes surrounding the brain and spinal cord. Most are benign (non-cancerous).",
                    "severity": "Low to Moderate",
                    "recommendation": "Many meningiomas are slow-growing and may only require monitoring. Symptomatic cases may need surgical intervention.",
                    "prevalence": "Most common primary brain tumor, about 36% of all brain tumors"
                },
                "notumor": {
                    "description": "No tumor detected in the brain MRI scan. The brain tissue appears normal.",
                    "severity": "None",
                    "recommendation": "Continue routine health monitoring. If symptoms persist, consult a neurologist for further evaluation.",
                    "prevalence": "N/A"
                },
                "pituitary": {
                    "description": "Pituitary tumors (adenomas) develop in the pituitary gland. Most are benign and can affect hormone production.",
                    "severity": "Low to Moderate",
                    "recommendation": "Endocrinological evaluation recommended. Treatment depends on tumor size and hormone activity.",
                    "prevalence": "About 10-15% of all intracranial tumors"
                }
            }
        },
        "pneumonia": {
            "name": "Chest X-Ray Pneumonia Detection",
            "description": "Detects pneumonia from chest X-ray images using AI analysis",
            "classes": ["Normal", "Pneumonia"],
            "color": "#06b6d4",
            "medical_details": {
                "NORMAL": {
                    "description": "Chest X-ray shows clear lung fields with no signs of infection or consolidation.",
                    "severity": "None",
                    "recommendation": "No immediate intervention required. Continue standard care if symptomatic.",
                    "prevalence": "N/A"
                },
                "PNEUMONIA": {
                    "description": "Pneumonia is a lung infection causing inflammation of the air sacs. X-ray shows opacities or consolidation patterns.",
                    "severity": "Moderate to High",
                    "recommendation": "Medical evaluation required. Treatment typically includes antibiotics for bacterial pneumonia. Hospitalization may be needed for severe cases.",
                    "prevalence": "Affects millions globally each year"
                }
            }
        },
        "retina": {
            "name": "Diabetic Retinopathy Screening",
            "description": "Grades diabetic retinopathy severity from fundus images",
            "classes": ["No DR", "Mild", "Moderate", "Severe", "Proliferative"],
            "color": "#10b981",
            "medical_details": {
                "No_DR": {
                    "description": "No signs of diabetic retinopathy detected. Retina appears healthy.",
                    "severity": "None",
                    "recommendation": "Continue annual diabetic eye exams. Maintain good blood sugar control.",
                    "prevalence": "N/A"
                },
                "Mild": {
                    "description": "Mild non-proliferative DR with microaneurysms (small areas of balloon-like swelling in the retina's blood vessels).",
                    "severity": "Low",
                    "recommendation": "Schedule follow-up in 6-12 months. Focus on glucose and blood pressure control.",
                    "prevalence": "Early stage affecting many diabetic patients"
                },
                "Moderate": {
                    "description": "Moderate non-proliferative DR with blocked blood vessels that nourish the retina.",
                    "severity": "Moderate",
                    "recommendation": "Ophthalmologist consultation within 3-6 months. Consider more aggressive diabetes management.",
                    "prevalence": "Progression indicator requiring attention"
                },
                "Severe": {
                    "description": "Severe non-proliferative DR with many blocked blood vessels, depriving areas of the retina of blood supply.",
                    "severity": "High",
                    "recommendation": "Urgent ophthalmologist referral. High risk of progressing to proliferative DR. May require laser treatment.",
                    "prevalence": "Significant vision loss risk"
                },
                "Proliferative_DR": {
                    "description": "Proliferative DR with new, abnormal blood vessel growth (neovascularization) that can bleed and cause vision loss.",
                    "severity": "Critical",
                    "recommendation": "Immediate ophthalmologist consultation. Treatment options include laser surgery, vitrectomy, or anti-VEGF injections.",
                    "prevalence": "Advanced stage with high risk of blindness"
                }
            }
        }
    }
    return info.get(disease_type, {})


def get_gradcam_interpretation() -> Dict[str, str]:
    """Get Grad-CAM color interpretation guide."""
    return {
        "title": "Grad-CAM Interpretation Guide",
        "description": "Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes which regions of the image most influenced the AI's decision.",
        "colors": {
            "Red/Yellow (Hot)": "High activation - Areas the model focused on most strongly for its prediction",
            "Green": "Moderate activation - Regions with some influence on the decision",
            "Blue/Purple (Cool)": "Low activation - Areas with minimal contribution to the prediction"
        },
        "clinical_note": "Hot regions typically indicate pathological features like lesions, opacities, or abnormal structures that the model identified as diagnostically significant.",
        "disclaimer": "Grad-CAM is an explainability tool and should be interpreted alongside clinical findings, not as a standalone diagnostic."
    }
