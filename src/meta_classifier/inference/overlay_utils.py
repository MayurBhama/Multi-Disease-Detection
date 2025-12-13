# src/meta_classifier/inference/overlay_utils.py
"""
Professional Overlay Utilities for Grad-CAM
============================================
Clean, minimal overlays for medical imaging.
"""

import numpy as np
import cv2
from typing import Tuple, Optional


# =====================================================
# COLORMAPS
# =====================================================
COLORMAPS = {
    "jet": cv2.COLORMAP_JET,
    "turbo": cv2.COLORMAP_TURBO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "inferno": cv2.COLORMAP_INFERNO,
    "plasma": cv2.COLORMAP_PLASMA,
    "hot": cv2.COLORMAP_HOT,
}


def apply_colormap(heatmap: np.ndarray, colormap: str = "turbo") -> np.ndarray:
    """Apply colormap to normalized heatmap (0-1). Returns RGB."""
    cm = COLORMAPS.get(colormap.lower(), cv2.COLORMAP_TURBO)
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    colored = cv2.applyColorMap(heatmap_uint8, cm)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def create_overlay(
    original: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "turbo"
) -> np.ndarray:
    """
    Create clean Grad-CAM overlay without any text.
    
    Args:
        original: Original RGB image
        heatmap: Normalized heatmap (0-1)
        alpha: Overlay transparency (0.3-0.5 recommended)
        colormap: Colormap name
        
    Returns:
        Clean overlay image (RGB)
    """
    h, w = original.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = apply_colormap(heatmap_resized, colormap)
    
    # Ensure uint8
    if original.dtype != np.uint8:
        if original.max() <= 1:
            original = (original * 255).astype(np.uint8)
        else:
            original = original.astype(np.uint8)
    
    # Blend
    overlay = cv2.addWeighted(original, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


def create_comparison(
    original: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    border_width: int = 2,
    border_color: Tuple[int, int, int] = (40, 40, 40)
) -> np.ndarray:
    """
    Create clean side-by-side comparison with minimal borders.
    
    No text labels - clean visual comparison only.
    """
    h, w = original.shape[:2]
    
    # Ensure same size and uint8
    heatmap = cv2.resize(heatmap, (w, h))
    overlay = cv2.resize(overlay, (w, h))
    
    for img in [original, heatmap, overlay]:
        if img.dtype != np.uint8:
            if img.max() <= 1:
                img = (img * 255).astype(np.uint8)
    
    # Create border
    border = np.full((h, border_width, 3), border_color, dtype=np.uint8)
    
    # Stack with borders
    combined = np.hstack([
        original,
        border.copy(),
        heatmap,
        border.copy(),
        overlay
    ])
    
    return combined


def create_annotated_overlay(
    original: np.ndarray,
    heatmap: np.ndarray,
    prediction: str,
    confidence: float,
    alpha: float = 0.4,
    colormap: str = "turbo"
) -> np.ndarray:
    """
    Create overlay with minimal, clean annotation.
    
    Uses small, unobtrusive corner label.
    """
    overlay = create_overlay(original, heatmap, alpha, colormap)
    h, w = overlay.shape[:2]
    
    # Scale font based on image size
    base_size = min(h, w)
    font_scale = max(0.4, base_size / 600)
    thickness = max(1, int(base_size / 300))
    
    # Create label - cap at 99% to avoid unrealistic 100% (medical AI best practice)
    capped_confidence = min(confidence, 0.99)
    label = f"{prediction} ({capped_confidence:.0%})"
    
    # Use cleaner font
    font = cv2.FONT_HERSHEY_DUPLEX
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Position in bottom-left corner with padding
    pad = int(base_size * 0.02)
    text_x = pad
    text_y = h - pad
    
    # Draw subtle background
    bg_pts = np.array([
        [0, h],
        [0, h - text_h - 2*pad],
        [text_w + 2*pad, h - text_h - 2*pad],
        [text_w + 3*pad, h]
    ])
    
    # Semi-transparent dark background
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [bg_pts], 255)
    
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask > 0,
            overlay[:, :, c] * 0.3,
            overlay[:, :, c]
        ).astype(np.uint8)
    
    # Draw text with anti-aliasing
    cv2.putText(
        overlay, label, (text_x, text_y),
        font, font_scale, (255, 255, 255), thickness,
        lineType=cv2.LINE_AA
    )
    
    return overlay


def create_professional_report(
    original: np.ndarray,
    heatmap_raw: np.ndarray,
    prediction: str,
    confidence: float,
    disease_type: str,
    colormap: str = "turbo",
    alpha: float = 0.4
) -> np.ndarray:
    """
    Create a professional report-style image.
    
    Layout: Original | Heatmap | Overlay
    With clean header showing prediction.
    """
    h, w = original.shape[:2]
    
    # Create colored heatmap
    heatmap = apply_colormap(cv2.resize(heatmap_raw, (w, h)), colormap)
    
    # Create overlay
    overlay = create_overlay(original, heatmap_raw, alpha, colormap)
    
    # Ensure uint8
    if original.dtype != np.uint8:
        original = (original * 255 if original.max() <= 1 else original).astype(np.uint8)
    
    # Panel dimensions
    panel_w = w
    total_w = panel_w * 3 + 4  # 2px borders
    header_h = max(30, int(h * 0.08))
    total_h = h + header_h
    
    # Create canvas with dark header
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    canvas[:header_h, :] = (30, 30, 35)
    
    # Place images
    y_start = header_h
    canvas[y_start:y_start+h, 0:panel_w] = original
    canvas[y_start:y_start+h, panel_w+2:2*panel_w+2] = heatmap
    canvas[y_start:y_start+h, 2*panel_w+4:] = overlay
    
    # Add thin separator lines
    canvas[y_start:, panel_w:panel_w+2] = (50, 50, 55)
    canvas[y_start:, 2*panel_w+2:2*panel_w+4] = (50, 50, 55)
    
    # Header text - clean and minimal
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = max(0.5, header_h / 50)
    thickness = max(1, int(header_h / 25))
    
    # Disease type on left
    disease_label = disease_type.replace("_", " ").title()
    cv2.putText(
        canvas, disease_label, (10, int(header_h * 0.7)),
        font, font_scale * 0.8, (150, 150, 155), thickness,
        lineType=cv2.LINE_AA
    )
    
    # Prediction on right - cap at 99% for realism
    capped_confidence = min(confidence, 0.99)
    pred_label = f"{prediction}: {capped_confidence:.1%}"
    (pred_w, _), _ = cv2.getTextSize(pred_label, font, font_scale, thickness)
    cv2.putText(
        canvas, pred_label, (total_w - pred_w - 10, int(header_h * 0.7)),
        font, font_scale, (255, 255, 255), thickness,
        lineType=cv2.LINE_AA
    )
    
    return canvas


# =====================================================
# LEGACY FUNCTIONS (for backward compatibility)
# =====================================================
def add_prediction_label(image, prediction, confidence, position="top", 
                         font_scale=0.7, thickness=2, bg_alpha=0.7):
    """Legacy function - returns annotated overlay."""
    return create_annotated_overlay(
        image, np.ones(image.shape[:2]) * 0.5,  # dummy heatmap
        prediction, confidence, alpha=0
    )


def add_confidence_bar(image, confidence, position="bottom", bar_height=8, margin=10):
    """Legacy - confidence bars removed for cleaner look."""
    return image


def create_side_by_side(original, heatmap, overlay, prediction, confidence, labels=None):
    """Legacy wrapper for clean comparison."""
    return create_comparison(original, heatmap, overlay)


def create_professional_overlay(original, heatmap, prediction, confidence,
                                 alpha=0.4, colormap="turbo", 
                                 add_label=True, add_bar=True, output_size=None):
    """
    Production overlay - clean design.
    """
    if add_label:
        result = create_annotated_overlay(original, heatmap, prediction, confidence, alpha, colormap)
    else:
        result = create_overlay(original, heatmap, alpha, colormap)
    
    if output_size:
        result = cv2.resize(result, output_size)
    
    return result
