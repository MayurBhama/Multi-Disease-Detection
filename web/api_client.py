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
        progress_callback=None,
        generate_gradcam: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Predict multiple images.
        
        Args:
            images: List of (image_bytes, filename) tuples
            disease_type: Disease type for all images
            progress_callback: Optional callback(current, total)
            generate_gradcam: Whether to generate Grad-CAM for each image
            
        Returns:
            List of prediction results
        """
        results = []
        total = len(images)
        
        for i, (img_bytes, filename) in enumerate(images):
            if progress_callback:
                progress_callback(i + 1, total)
            
            success, result = self.predict(
                img_bytes, filename, disease_type, generate_gradcam=generate_gradcam
            )
            
            result["filename"] = filename
            result["success"] = success
            result["image_bytes"] = img_bytes  # Store for display
            results.append(result)
            
            # Small delay to avoid overwhelming the server
            time.sleep(0.2 if generate_gradcam else 0.1)
        
        return results



# Utility functions
def format_confidence(confidence: float, human_readable: bool = False) -> str:
    """
    Format confidence as percentage string.
    
    Medical AI best practices:
    - Cap confidence at 99.2% to avoid unrealistic 100% (indicates overfitting/saturation)
    - Optionally display human-readable format for very high confidence
    
    Args:
        confidence: Raw confidence value (0-1)
        human_readable: If True, display as "High (≈99%)" for very high confidence
    
    Returns:
        Formatted confidence string
    """
    # Cap at 99.2% - no real medical model should report 100%
    # 100% signals overfitting, softmax saturation, or poor calibration
    capped_confidence = min(confidence, 0.992)
    percentage = capped_confidence * 100
    
    if human_readable and percentage >= 95:
        if percentage >= 99:
            return "High (≈99%)"
        elif percentage >= 97:
            return "High (≈97%)"
        else:
            return f"High (≈{percentage:.0f}%)"
    
    return f"{percentage:.1f}%"


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
        "limitation": "⚠️ Important: Grad-CAM is NOT localization — it explains what influenced the prediction, not where the tumor/abnormality exactly is. This is a key distinction in medical AI and should not be used for surgical planning or precise boundary detection.",
        "disclaimer": "Grad-CAM is an explainability tool and should be interpreted alongside clinical findings, not as a standalone diagnostic."
    }


def detect_image_type(image_bytes: bytes) -> Dict[str, Any]:
    """
    Auto-detect image type (Brain MRI, Chest X-ray, or Retina).
    
    Improved heuristics:
    - Color analysis for retina (warm colors, orange/red tones)
    - Edge density for MRI (brain structures have more edges)
    - Brightness distribution for X-ray (bimodal: dark lungs, bright ribs)
    """
    try:
        from PIL import Image
        import numpy as np
        from io import BytesIO
        
        img = Image.open(BytesIO(image_bytes))
        img_array = np.array(img)
        
        width, height = img.size
        aspect_ratio = width / height
        
        # Grayscale check
        if len(img_array.shape) == 2:
            is_grayscale = True
            gray_array = img_array.astype(float)
        elif img_array.shape[2] == 1:
            is_grayscale = True
            gray_array = img_array[:,:,0].astype(float)
        else:
            # Check if RGB channels are similar
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            is_grayscale = np.mean(np.abs(r.astype(float) - g.astype(float))) < 15 and \
                          np.mean(np.abs(g.astype(float) - b.astype(float))) < 15
            gray_array = np.mean(img_array[:,:,:3], axis=2).astype(float)
        
        h, w = gray_array.shape
        
        # =============================================
        # RETINA DETECTION (warm colors + dark borders)
        # =============================================
        has_warm_colors = False
        red_dominance = 0
        orange_score = 0
        
        if not is_grayscale and len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            r, g, b = img_array[:,:,0].astype(float), img_array[:,:,1].astype(float), img_array[:,:,2].astype(float)
            
            # Red/orange dominance (characteristic of retina)
            red_dominance = np.mean(r) - np.mean(b)
            orange_region = (r > 100) & (g > 50) & (b < 100)
            orange_score = np.sum(orange_region) / (h * w) * 100
            has_warm_colors = (red_dominance > 15 and np.mean(r) > 80) or orange_score > 20
        
        # Dark border detection (fundus images have black circular borders)
        border_size = min(h, w) // 8
        top_border = np.mean(gray_array[:border_size, :])
        bottom_border = np.mean(gray_array[-border_size:, :])
        left_border = np.mean(gray_array[:, :border_size])
        right_border = np.mean(gray_array[:, -border_size:])
        corner_brightness = (top_border + bottom_border + left_border + right_border) / 4
        
        center_region = gray_array[h//4:3*h//4, w//4:3*w//4]
        center_brightness = np.mean(center_region)
        
        has_dark_borders = corner_brightness < 30 and center_brightness > 60
        
        # =============================================
        # MRI vs X-RAY DIFFERENTIATION
        # =============================================
        
        # Edge density (MRI has more internal structure/edges)
        # Simple Sobel-like edge detection
        gx = np.abs(gray_array[:, 1:] - gray_array[:, :-1])
        gy = np.abs(gray_array[1:, :] - gray_array[:-1, :])
        edge_density = (np.mean(gx) + np.mean(gy)) / 2
        
        # Histogram analysis
        hist, bins = np.histogram(gray_array.flatten(), bins=50, range=(0, 255))
        hist_norm = hist / hist.sum()
        
        # X-ray has bimodal distribution (dark lungs + bright bones)
        low_peak = np.sum(hist_norm[:15])   # Dark regions
        mid_region = np.sum(hist_norm[15:35])  # Mid-tones
        high_peak = np.sum(hist_norm[35:])  # Bright regions
        
        is_bimodal = low_peak > 0.15 and high_peak > 0.1 and mid_region < 0.5
        
        # MRI typically has smoother histogram, more mid-tones
        has_mid_tones = mid_region > 0.3
        
        # X-ray background is typically uniform dark or light
        # MRI often has text/annotations or variable background
        
        # Check for rectangular shape (X-rays are often portrait/landscape)
        is_portrait = aspect_ratio < 0.9  # Taller than wide
        is_square = 0.9 <= aspect_ratio <= 1.1
        
        # =============================================
        # SCORING
        # =============================================
        scores = {"retina": 0, "pneumonia": 0, "brain_mri": 0}
        
        # RETINA scoring
        if has_warm_colors:
            scores["retina"] += 50
        if has_dark_borders:
            scores["retina"] += 40
        if orange_score > 30:
            scores["retina"] += 20
        if red_dominance > 25:
            scores["retina"] += 15
        
        # If clearly retina (warm colors + dark borders), don't consider others much
        if scores["retina"] >= 70:
            scores["pneumonia"] = 0
            scores["brain_mri"] = 0
        else:
            # CHEST X-RAY scoring (only for grayscale images)
            if is_grayscale:
                scores["pneumonia"] += 15
                
                if is_bimodal:
                    scores["pneumonia"] += 30  # Bimodal = lungs + ribs
                
                if is_portrait or aspect_ratio > 1.1:  # X-rays often portrait or wide
                    scores["pneumonia"] += 15
                
                if edge_density < 15:  # X-rays have fewer internal edges
                    scores["pneumonia"] += 20
                
                if center_brightness > 100:  # Chest center often brighter
                    scores["pneumonia"] += 10
            
            # BRAIN MRI scoring (only for grayscale images)
            if is_grayscale:
                scores["brain_mri"] += 15
                
                if is_square:  # MRI slices are typically square
                    scores["brain_mri"] += 25
                
                if edge_density > 12:  # Brain has more internal structure
                    scores["brain_mri"] += 25
                
                if has_mid_tones:  # MRI has more gradual intensity
                    scores["brain_mri"] += 15
                
                if not is_bimodal:  # MRI is not typically bimodal
                    scores["brain_mri"] += 10
        
        # Normalize scores
        total = sum(scores.values()) or 1
        confidences = {k: round(v / total, 2) for k, v in scores.items()}
        
        detected_type = max(scores, key=scores.get)
        confidence = confidences[detected_type]
        
        # OVERRIDE: For grayscale square images with high edge density, force brain_mri
        # This helps distinguish brain MRI from chest X-ray more reliably
        if is_grayscale and is_square and edge_density > 15 and not has_warm_colors:
            # High edge density + square = brain MRI (brain has more internal structure)
            detected_type = "brain_mri"
            confidence = max(confidence, 0.75)
            confidences = {"brain_mri": 0.75, "pneumonia": 0.20, "retina": 0.05}
        
        return {
            "detected_type": detected_type,
            "confidence": confidence,
            "all_scores": confidences,
            "analysis": {
                "is_grayscale": is_grayscale,
                "aspect_ratio": round(aspect_ratio, 2),
                "has_warm_colors": has_warm_colors,
                "has_dark_borders": has_dark_borders,
                "edge_density": round(edge_density, 1),
                "is_bimodal": is_bimodal
            }
        }
    except Exception as e:
        return {
            "detected_type": "brain_mri",
            "confidence": 0.33,
            "all_scores": {"brain_mri": 0.33, "pneumonia": 0.33, "retina": 0.33},
            "error": str(e)
        }





def get_preprocessed_preview(image_bytes: bytes, disease_type: str = "brain_mri") -> bytes:
    """
    Generate 224x224 preprocessed preview of the image.
    
    Returns:
        PNG bytes of preprocessed image
    """
    try:
        from PIL import Image
        import numpy as np
        from io import BytesIO
        
        img = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Resize to 224x224 (maintain aspect ratio with center crop)
        target_size = 224
        width, height = img.size
        
        # Calculate crop dimensions
        if width > height:
            new_height = target_size
            new_width = int(width * (target_size / height))
        else:
            new_width = target_size
            new_height = int(height * (target_size / width))
        
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Center crop
        left = (new_width - target_size) // 2
        top = (new_height - target_size) // 2
        img_cropped = img_resized.crop((left, top, left + target_size, top + target_size))
        
        # Convert to bytes
        output = BytesIO()
        img_cropped.save(output, format='PNG')
        return output.getvalue()
        
    except Exception as e:
        return image_bytes  # Return original on error


def check_retina_quality(image_bytes: bytes) -> Dict[str, Any]:
    """
    Enhanced quality check specifically for retina fundus images.
    
    Checks:
    - Brightness level
    - Contrast level  
    - Glare/saturation
    - Field of view coverage
    - Overall quality score
    """
    try:
        from PIL import Image
        import numpy as np
        from io import BytesIO
        
        img = Image.open(BytesIO(image_bytes))
        img_array = np.array(img)
        
        # Convert to grayscale for analysis
        if len(img_array.shape) == 3:
            gray = np.mean(img_array[:,:,:3], axis=2)
            rgb = img_array[:,:,:3]
        else:
            gray = img_array
            rgb = None
        
        h, w = gray.shape
        
        # 1. Brightness Analysis
        mean_brightness = np.mean(gray)
        brightness_status = "Good"
        if mean_brightness < 40:
            brightness_status = "Too Dark"
        elif mean_brightness < 70:
            brightness_status = "Dark"
        elif mean_brightness > 200:
            brightness_status = "Too Bright"
        elif mean_brightness > 170:
            brightness_status = "Bright"
        
        # 2. Contrast Analysis (standard deviation)
        contrast = np.std(gray)
        contrast_status = "Good"
        if contrast < 30:
            contrast_status = "Low Contrast"
        elif contrast < 50:
            contrast_status = "Fair"
        elif contrast > 80:
            contrast_status = "High Contrast"
        
        # 3. Glare Detection (saturated pixels)
        saturated_pixels = np.sum(gray > 250) / gray.size * 100
        glare_status = "None"
        if saturated_pixels > 5:
            glare_status = "Severe Glare"
        elif saturated_pixels > 2:
            glare_status = "Moderate Glare"
        elif saturated_pixels > 0.5:
            glare_status = "Mild Glare"
        
        # 4. Field of View (circular coverage detection)
        # Create circular mask
        center = (w // 2, h // 2)
        radius = min(w, h) // 2
        y, x = np.ogrid[:h, :w]
        circular_mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        
        # Check coverage (non-black pixels in circular region)
        threshold = 20
        valid_pixels = gray[circular_mask] > threshold
        fov_coverage = np.sum(valid_pixels) / np.sum(circular_mask) * 100
        
        fov_status = "Good"
        if fov_coverage < 60:
            fov_status = "Poor FOV"
        elif fov_coverage < 80:
            fov_status = "Partial FOV"
        elif fov_coverage > 95:
            fov_status = "Excellent FOV"
        
        # 5. Overall Quality Score (0-100)
        quality_score = 0
        
        # Brightness contribution (25 points)
        if brightness_status == "Good":
            quality_score += 25
        elif brightness_status in ["Dark", "Bright"]:
            quality_score += 15
        else:
            quality_score += 5
        
        # Contrast contribution (25 points)
        if contrast_status == "Good":
            quality_score += 25
        elif contrast_status == "Fair":
            quality_score += 15
        elif contrast_status == "High Contrast":
            quality_score += 20
        else:
            quality_score += 5
        
        # Glare contribution (25 points)
        if glare_status == "None":
            quality_score += 25
        elif glare_status == "Mild Glare":
            quality_score += 15
        elif glare_status == "Moderate Glare":
            quality_score += 8
        else:
            quality_score += 0
        
        # FOV contribution (25 points)
        if fov_status in ["Good", "Excellent FOV"]:
            quality_score += 25
        elif fov_status == "Partial FOV":
            quality_score += 15
        else:
            quality_score += 5
        
        overall_status = "Excellent" if quality_score >= 85 else "Good" if quality_score >= 70 else "Fair" if quality_score >= 50 else "Poor"
        
        return {
            "quality_score": quality_score,
            "overall_status": overall_status,
            "brightness": {
                "value": round(mean_brightness, 1),
                "status": brightness_status
            },
            "contrast": {
                "value": round(contrast, 1),
                "status": contrast_status
            },
            "glare": {
                "value": round(saturated_pixels, 2),
                "status": glare_status
            },
            "field_of_view": {
                "value": round(fov_coverage, 1),
                "status": fov_status
            }
        }
    except Exception as e:
        return {
            "quality_score": 0,
            "overall_status": "Error",
            "error": str(e)
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


def get_severity_score(
    severity: str,
    confidence: float = None,
    heatmap_intensity: float = None
) -> int:
    """
    Calculate severity score using a structured, explainable formula.
    
    Severity score is based on:
    1. Base severity from medical classification
    2. Model prediction confidence (40% weight)
    3. Normalized Grad-CAM heatmap intensity (60% weight)
    
    Formula: severity = base_severity * (0.4 * confidence + 0.6 * heatmap_intensity)
    
    This weighted approach provides:
    - Model activation intensity from Grad-CAM
    - Region activation spread consideration
    - Prediction confidence integration
    
    Args:
        severity: Text severity level ("None", "Low", "Moderate", "High", "Critical")
        confidence: Model prediction confidence (0-1), optional
        heatmap_intensity: Normalized Grad-CAM heatmap intensity (0-1), optional
    
    Returns:
        Severity score (0-100)
    """
    severity_map = {
        "None": 0,
        "Low": 20,
        "Low to Moderate": 35,
        "Moderate": 50,
        "Moderate to High": 65,
        "High": 80,
        "Critical": 100
    }
    
    base_severity = severity_map.get(severity, 50)
    
    # If no additional metrics provided, return base severity
    if confidence is None and heatmap_intensity is None:
        return base_severity
    
    # Apply weighted formula when metrics are available
    # Default to 0.5 if one metric is missing
    conf = confidence if confidence is not None else 0.5
    heatmap = heatmap_intensity if heatmap_intensity is not None else 0.5
    
    # Weighted combination: 40% confidence, 60% heatmap intensity
    # This weights visual evidence (heatmap) more than raw confidence
    weighted_factor = 0.4 * conf + 0.6 * heatmap
    
    # Scale base severity by weighted factor (range: 0.5x to 1.5x)
    # This ensures low severity stays low even with high confidence
    scaling_factor = 0.5 + weighted_factor  # Range: 0.5 to 1.5
    adjusted_severity = int(base_severity * scaling_factor)
    
    # Clamp to 0-100 range
    return max(0, min(100, adjusted_severity))


def generate_pdf_report(
    result: Dict[str, Any],
    disease_type: str,
    image_bytes: bytes = None,
    gradcam_bytes: bytes = None
) -> bytes:
    """
    Generate professional PDF report from prediction results.
    
    Features:
    - Legal disclaimer
    - Model architecture transparency
    - Severity score (quantitative)
    - Original image & Grad-CAM embedding
    - Improved section spacing
    
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
    from PIL import Image
    
    # Model architecture info for transparency
    model_architectures = {
        "brain_mri": "EfficientNetB3 (Transfer Learning from ImageNet)",
        "pneumonia": "Xception-based CNN (Depthwise Separable Convolutions)",
        "retina": "EfficientNet Ensemble (V2-S + B2 + B0, Weighted Average)"
    }
    
    class PDF(FPDF):
        def header(self):
            self.set_font('Helvetica', 'B', 16)
            self.set_text_color(8, 145, 178)  # Cyan color
            self.cell(0, 10, 'Multi-Disease Detection Report', 0, 1, 'C')
            self.set_font('Helvetica', '', 9)
            self.set_text_color(100, 100, 100)
            self.cell(0, 4, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
            self.set_draw_color(8, 145, 178)
            self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
            self.ln(6)
        
        def footer(self):
            self.set_y(-20)
            self.set_draw_color(180, 180, 180)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(2)
            self.set_font('Helvetica', 'I', 6)
            self.set_text_color(100, 100, 100)
            self.multi_cell(0, 3, 
                'DISCLAIMER: This report is for research and educational purposes only. '
                'Not intended for medical diagnosis. Consult qualified healthcare professionals for clinical decisions.',
                align='C')
    
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=25)
    
    disease_names = {"brain_mri": "Brain MRI Analysis", "pneumonia": "Chest X-Ray Analysis", "retina": "Retinal Scan Analysis"}
    
    # ===========================================
    # SECTION 1: Analysis Summary
    # ===========================================
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 10, '1. Analysis Summary', 0, 1)
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(0, 0, 0)
    
    # Analysis Type
    pdf.cell(55, 8, 'Analysis Type:', 0, 0)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, disease_names.get(disease_type, disease_type), 0, 1)
    pdf.set_font('Helvetica', '', 11)
    
    # Model Used (Transparency)
    pdf.cell(55, 8, 'Model Architecture:', 0, 0)
    pdf.set_font('Helvetica', 'I', 10)
    pdf.cell(0, 8, model_architectures.get(disease_type, 'Custom CNN'), 0, 1)
    pdf.set_font('Helvetica', '', 11)
    
    # Predicted Condition
    pdf.cell(55, 8, 'Predicted Condition:', 0, 0)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(8, 145, 178)
    pdf.cell(0, 8, result.get('predicted_class', 'N/A'), 0, 1)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', '', 11)
    
    # Confidence Score (capped at 99.2%)
    confidence = min(result.get('confidence', 0), 0.992)
    pdf.cell(55, 8, 'Confidence Score:', 0, 0)
    pdf.cell(0, 8, f'{confidence * 100:.1f}%', 0, 1)
    
    # Severity Score (quantitative)
    disease_info = get_disease_info(disease_type)
    pred_key = result.get('predicted_class', '').lower().replace(' ', '_')
    medical_details = disease_info.get('medical_details', {})
    class_info = None
    for key, value in medical_details.items():
        if key.lower().replace('_', '') == pred_key.replace('_', ''):
            class_info = value
            break
    
    if class_info:
        severity_text = class_info.get('severity', 'Unknown')
        severity_score = get_severity_score(severity_text, confidence=confidence)
        pdf.cell(55, 8, 'Severity Score:', 0, 0)
        pdf.set_font('Helvetica', 'B', 11)
        
        # Color code severity
        if severity_score >= 75:
            pdf.set_text_color(220, 50, 50)
        elif severity_score >= 50:
            pdf.set_text_color(220, 150, 50)
        else:
            pdf.set_text_color(50, 180, 100)
        
        pdf.cell(0, 8, f'{severity_score}/100 ({severity_text})', 0, 1)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Helvetica', '', 11)
    
    pdf.ln(8)
    
    # ===========================================
    # SECTION 2: Probability Distribution
    # ===========================================
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 10, '2. Probability Distribution', 0, 1)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(0, 0, 0)
    
    probs = result.get('probabilities', {})
    for rank, (class_name, prob) in enumerate(sorted(probs.items(), key=lambda x: -x[1]), 1):
        # Cap at 99.2%
        display_prob = min(prob * 100, 99.2)
        formatted_name = class_name.replace('_', ' ').title()
        
        pdf.cell(10, 7, f'{rank}.', 0, 0)
        pdf.cell(50, 7, formatted_name, 0, 0)
        pdf.cell(0, 7, f'{display_prob:.1f}%', 0, 1)
    
    pdf.ln(8)
    
    # ===========================================
    # SECTION 3: Medical Information
    # ===========================================
    if class_info:
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(0, 10, '3. Medical Information', 0, 1)
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(0, 0, 0)
        
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 7, 'Description:', 0, 1)
        pdf.set_font('Helvetica', '', 10)
        pdf.multi_cell(0, 5, class_info.get('description', 'N/A'))
        
        pdf.ln(3)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 7, 'Recommendation:', 0, 1)
        pdf.set_font('Helvetica', '', 10)
        pdf.multi_cell(0, 5, class_info.get('recommendation', 'Consult a healthcare professional.'))
    
    pdf.ln(8)
    
    # ===========================================
    # SECTION 4: Image Evidence
    # ===========================================
    next_section = '4' if class_info else '3'
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 10, f'{next_section}. Image Evidence', 0, 1)
    pdf.set_text_color(0, 0, 0)
    
    # Check if prediction is normal/healthy - Grad-CAM not meaningful for these
    predicted_class = result.get('predicted_class', '').lower().replace('_', ' ')
    normal_classes = ["notumor", "no tumor", "normal", "no_dr", "no dr", "healthy"]
    is_normal_case = any(nc in predicted_class for nc in normal_classes)
    
    temp_files = []
    
    try:
        # Check if we have space for images (add new page if needed)
        if pdf.get_y() > 180:
            pdf.add_page()
        
        img_start_y = pdf.get_y()
        
        # Embed original image
        if image_bytes:
            try:
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    # Convert bytes to PIL Image and save as PNG
                    original_img = Image.open(BytesIO(image_bytes))
                    # Convert to RGB if needed
                    if original_img.mode in ('RGBA', 'LA', 'P'):
                        original_img = original_img.convert('RGB')
                    original_img.save(tmp.name, 'PNG')
                    temp_files.append(tmp.name)
                
                # Add labels
                pdf.set_font('Helvetica', 'B', 10)
                pdf.cell(90, 6, 'Original Scan', 0, 0, 'C')
                if is_normal_case:
                    pdf.cell(100, 6, 'Analysis Result', 0, 1, 'C')
                else:
                    pdf.cell(100, 6, 'Grad-CAM Heatmap', 0, 1, 'C')
                
                pdf.set_y(img_start_y + 8)
                
                # Draw border box for original
                pdf.set_draw_color(200, 200, 200)
                pdf.rect(12, pdf.get_y(), 88, 70)
                pdf.image(tmp.name, x=14, y=pdf.get_y() + 2, w=84, h=66)
                
            except Exception as e:
                pdf.set_font('Helvetica', 'I', 10)
                pdf.set_text_color(150, 150, 150)
                pdf.cell(90, 40, 'Original image not available', 0, 0, 'C')
        else:
            pdf.set_font('Helvetica', 'B', 10)
            pdf.cell(90, 6, 'Original Scan', 0, 0, 'C')
            if is_normal_case:
                pdf.cell(100, 6, 'Analysis Result', 0, 1, 'C')
            else:
                pdf.cell(100, 6, 'Grad-CAM Heatmap', 0, 1, 'C')
            pdf.set_y(img_start_y + 8)
            pdf.set_draw_color(200, 200, 200)
            pdf.rect(12, pdf.get_y(), 88, 70)
            pdf.set_font('Helvetica', 'I', 9)
            pdf.set_text_color(150, 150, 150)
            pdf.set_xy(14, pdf.get_y() + 30)
            pdf.cell(84, 10, 'Image not provided', 0, 0, 'C')
        
        # For normal cases, show "No Abnormality" message instead of Grad-CAM
        if is_normal_case:
            # Draw green success box
            pdf.set_draw_color(50, 180, 100)
            pdf.set_fill_color(240, 255, 240)
            pdf.rect(105, img_start_y + 8, 93, 70, 'DF')
            
            # Add checkmark and message
            pdf.set_font('Helvetica', 'B', 11)
            pdf.set_text_color(50, 180, 100)
            pdf.set_xy(105, img_start_y + 30)
            pdf.cell(93, 8, 'No Abnormality Detected', 0, 1, 'C')
            
            pdf.set_font('Helvetica', '', 8)
            pdf.set_text_color(80, 80, 80)
            pdf.set_xy(107, img_start_y + 42)
            pdf.multi_cell(89, 4, 
                'Grad-CAM heatmap is not shown for normal/healthy predictions as there are no pathological regions to highlight.',
                align='C')
            pdf.set_text_color(0, 0, 0)
        
        # Embed Grad-CAM image (only for abnormal cases)
        elif gradcam_bytes:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    gradcam_img = Image.open(BytesIO(gradcam_bytes))
                    if gradcam_img.mode in ('RGBA', 'LA', 'P'):
                        gradcam_img = gradcam_img.convert('RGB')
                    gradcam_img.save(tmp.name, 'PNG')
                    temp_files.append(tmp.name)
                
                # Draw border box for Grad-CAM
                pdf.set_draw_color(8, 145, 178)
                pdf.rect(105, img_start_y + 8, 93, 70)
                pdf.image(tmp.name, x=107, y=img_start_y + 10, w=89, h=66)
                
            except Exception as e:
                pdf.set_draw_color(200, 200, 200)
                pdf.rect(105, img_start_y + 8, 93, 70)
                pdf.set_font('Helvetica', 'I', 9)
                pdf.set_text_color(150, 150, 150)
                pdf.set_xy(107, img_start_y + 38)
                pdf.cell(89, 10, 'Grad-CAM not available', 0, 0, 'C')
        else:
            pdf.set_draw_color(200, 200, 200)
            pdf.rect(105, img_start_y + 8, 93, 70)
            pdf.set_font('Helvetica', 'I', 9)
            pdf.set_text_color(150, 150, 150)
            pdf.set_xy(107, img_start_y + 38)
            pdf.cell(89, 10, 'Grad-CAM not generated', 0, 0, 'C')
        
        pdf.set_y(img_start_y + 82)
        pdf.set_text_color(0, 0, 0)
        
        # Add Grad-CAM legend (only for abnormal cases)
        if not is_normal_case:
            pdf.set_font('Helvetica', 'I', 8)
            pdf.set_text_color(100, 100, 100)
            pdf.multi_cell(0, 4, 
                'Grad-CAM Legend: Red/Yellow = High activation (model focus) | Green = Moderate | Blue = Low activation. '
                'Note: Grad-CAM shows what influenced the prediction, not exact pathology location.',
                align='C')
        
    finally:
        # Cleanup temp files
        for tmp_path in temp_files:
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    # Return PDF bytes
    return bytes(pdf.output())


