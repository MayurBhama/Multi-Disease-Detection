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


def check_image_quality(image_bytes: bytes) -> Dict[str, Any]:
    """
    Check image quality before analysis.
    
    Returns dict with:
        - is_valid: bool
        - issues: list of detected issues
        - resolution: (width, height)
        - file_size_mb: float
        - blur_score: float (higher = sharper)
    """
    try:
        from PIL import Image
        import numpy as np
        from io import BytesIO
        
        # Load image
        img = Image.open(BytesIO(image_bytes))
        width, height = img.size
        
        issues = []
        
        # Check resolution
        if width < 100 or height < 100:
            issues.append("Image resolution too low (min 100x100)")
        elif width < 200 or height < 200:
            issues.append("Low resolution may affect accuracy")
        
        if width > 4000 or height > 4000:
            issues.append("Very high resolution - may slow processing")
        
        # Check file size
        file_size_mb = len(image_bytes) / (1024 * 1024)
        if file_size_mb > 10:
            issues.append("File size exceeds 10MB limit")
        
        # Check blur (Laplacian variance)
        blur_score = 0
        try:
            gray = img.convert("L")
            gray_array = np.array(gray, dtype=np.float32)
            
            # Laplacian kernel
            laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
            from scipy.ndimage import convolve
            laplacian_img = convolve(gray_array, laplacian)
            blur_score = float(laplacian_img.var())
            
            if blur_score < 100:
                issues.append("Image appears blurry")
            elif blur_score < 500:
                issues.append("Image may be slightly blurry")
        except ImportError:
            # scipy not available, skip blur check
            blur_score = -1
        except Exception:
            blur_score = -1
        
        # Check if grayscale or color
        mode = img.mode
        
        return {
            "is_valid": len([i for i in issues if "too low" in i or "exceeds" in i]) == 0,
            "issues": issues,
            "resolution": (width, height),
            "file_size_mb": round(file_size_mb, 2),
            "blur_score": round(blur_score, 2) if blur_score >= 0 else None,
            "mode": mode
        }
    except Exception as e:
        return {
            "is_valid": False,
            "issues": [f"Cannot read image: {str(e)}"],
            "resolution": (0, 0),
            "file_size_mb": 0,
            "blur_score": None,
            "mode": None
        }


def get_severity_score(severity: str) -> int:
    """Convert severity text to numeric score (0-100)."""
    severity_map = {
        "None": 0,
        "Low": 20,
        "Low to Moderate": 35,
        "Moderate": 50,
        "Moderate to High": 65,
        "High": 80,
        "Critical": 100
    }
    return severity_map.get(severity, 50)


def generate_pdf_report(
    result: Dict[str, Any],
    disease_type: str,
    image_bytes: bytes = None,
    gradcam_bytes: bytes = None
) -> bytes:
    """
    Generate PDF report from prediction results.
    
    Args:
        result: Prediction result dict
        disease_type: Type of analysis
        image_bytes: Original image (optional)
        gradcam_bytes: Grad-CAM image (optional)
    
    Returns:
        PDF bytes
    """
    from fpdf import FPDF
    from datetime import datetime
    from io import BytesIO
    import tempfile
    import os
    
    class PDF(FPDF):
        def header(self):
            self.set_font('Helvetica', 'B', 16)
            self.cell(0, 10, 'Multi-Disease Detection Report', 0, 1, 'C')
            self.set_font('Helvetica', '', 10)
            self.cell(0, 5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
            self.ln(5)
        
        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.cell(0, 10, 'DISCLAIMER: For research purposes only. Not a medical diagnosis.', 0, 0, 'C')
    
    pdf = PDF()
    pdf.add_page()
    
    # Analysis Summary
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Analysis Summary', 0, 1)
    pdf.set_font('Helvetica', '', 11)
    
    disease_names = {"brain_mri": "Brain MRI", "pneumonia": "Chest X-Ray", "retina": "Retinal Scan"}
    
    pdf.cell(60, 8, 'Analysis Type:', 0, 0)
    pdf.cell(0, 8, disease_names.get(disease_type, disease_type), 0, 1)
    
    pdf.cell(60, 8, 'Predicted Condition:', 0, 0)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, result.get('predicted_class', 'N/A'), 0, 1)
    pdf.set_font('Helvetica', '', 11)
    
    confidence = result.get('confidence', 0)
    pdf.cell(60, 8, 'Confidence Score:', 0, 0)
    pdf.cell(0, 8, f'{confidence * 100:.1f}%', 0, 1)
    
    pdf.ln(5)
    
    # Probability Distribution
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Probability Distribution', 0, 1)
    pdf.set_font('Helvetica', '', 10)
    
    probs = result.get('probabilities', {})
    for class_name, prob in sorted(probs.items(), key=lambda x: -x[1]):
        pdf.cell(60, 7, class_name, 0, 0)
        pdf.cell(0, 7, f'{prob * 100:.1f}%', 0, 1)
    
    pdf.ln(5)
    
    # Medical Information
    disease_info = get_disease_info(disease_type)
    pred_key = result.get('predicted_class', '').lower().replace(' ', '_')
    medical_details = disease_info.get('medical_details', {})
    
    class_info = None
    for key, value in medical_details.items():
        if key.lower().replace('_', '') == pred_key.replace('_', ''):
            class_info = value
            break
    
    if class_info:
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Medical Information', 0, 1)
        pdf.set_font('Helvetica', '', 10)
        
        pdf.cell(40, 8, 'Severity:', 0, 0)
        pdf.cell(0, 8, class_info.get('severity', 'Unknown'), 0, 1)
        
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 8, 'Description:', 0, 1)
        pdf.set_font('Helvetica', '', 10)
        pdf.multi_cell(0, 6, class_info.get('description', 'N/A'))
        
        pdf.ln(3)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 8, 'Recommendation:', 0, 1)
        pdf.set_font('Helvetica', '', 10)
        pdf.multi_cell(0, 6, class_info.get('recommendation', 'Consult a healthcare professional.'))
    
    # Return PDF bytes
    return pdf.output()
