"""
src/preprocessing/retina_preprocessing.py

Stable, mild preprocessing pipeline for diabetic retinopathy classification.
Goal: enhance lesion visibility without destroying subtle features (esp. class 1-2).

Main pipeline:
 - Safe read
 - Crop black borders
 - Detect circular fundus & crop (fallback to center crop)
 - Resize (keep aspect, pad)
 - Optional light Ben Graham (mild)
 - CLAHE (green channel or L-channel)
 - Optional gentle unsharp mask
 - Normalize to [0,1] (float32) in RGB order
"""

import os
import sys
import cv2
import numpy as np

# Import logging and exception handling
from src.utils.logger import get_logger
from src.utils.exception import PreprocessingError

# Initialize logger
logger = get_logger(__name__)

# ---------------------------
# Utility / I/O
# ---------------------------
def _read_image_bgr(path):
    """Read image robustly. Accepts bytes or str. Returns BGR image or None."""
    if isinstance(path, bytes):
        path = path.decode("utf-8")
    path = str(path)
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img

# ---------------------------
# Cropping helpers
# ---------------------------
def crop_black_borders(img, threshold=10, min_area_frac=0.01):
    """
    Remove large black borders. Returns cropped BGR image.
    threshold: pixel intensity threshold for mask
    """
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # Morphological closing to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    largest = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(largest)
    # if bounding box too small relative to image, return original
    if w*h < min_area_frac * img.shape[0]*img.shape[1]:
        return img
    # add small margin
    m = 4
    x = max(0, x-m); y = max(0, y-m)
    w = min(img.shape[1]-x, w+2*m); h = min(img.shape[0]-y, h+2*m)
    return img[y:y+h, x:x+w]

def detect_and_crop_circle(img, blur_ksize=5, thresh_val=7):
    """
    Detect circular fundus region and crop to circle bounding box.
    Returns BGR image.
    """
    if img is None:
        return None
    h,w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
    # morphological open to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # fallback center square
        size = min(h,w)
        cy, cx = h//2, w//2
        half = size//2
        return img[max(0,cy-half):min(h,cy+half), max(0,cx-half):min(w,cx+half)]
    largest = max(contours, key=cv2.contourArea)
    x,y,ww,hh = cv2.boundingRect(largest)
    cx = x + ww//2
    cy = y + hh//2
    radius = int(max(ww, hh)/2 * 1.02)  # small margin
    # mask & bitwise
    mask = np.zeros((h,w), dtype=np.uint8)
    cv2.circle(mask, (cx,cy), radius, 255, -1)
    masked = cv2.bitwise_and(img, img, mask=mask)
    y1 = max(0, cy-radius); y2 = min(h, cy+radius)
    x1 = max(0, cx-radius); x2 = min(w, cx+radius)
    cropped = masked[y1:y2, x1:x2]
    if cropped.size == 0:
        return img
    return cropped

# ---------------------------
# Resize / padding
# ---------------------------
def resize_maintain_aspect(img, target_size=(512,512), interp=cv2.INTER_AREA):
    """Resize preserving aspect ratio and pad with black background. Input BGR."""
    if img is None:
        return None
    if isinstance(target_size, (list,tuple)) and len(target_size)==2:
        th, tw = target_size
    else:
        raise ValueError("target_size must be (h,w)")
    h,w = img.shape[:2]
    if h==0 or w==0:
        return None
    scale = min(th/h, tw/w)
    new_h = max(1, int(round(h*scale)))
    new_w = max(1, int(round(w*scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    pad_top = (th - new_h)//2
    pad_bottom = th - new_h - pad_top
    pad_left = (tw - new_w)//2
    pad_right = tw - new_w - pad_left
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
    return padded

# Backwards-compatible alias expected by debug script
def resize_image(img, target_size=(512,512)):
    return resize_maintain_aspect(img, target_size=target_size)

def remove_border(img):
    return crop_black_borders(img)

# ---------------------------
# Enhancement: mild Ben-Graham (optional)
# ---------------------------
def ben_graham_mild(img, sigma=5, alpha=2.0):
    """
    Mild variant of Ben Graham: small sigma and lower weights so we don't destroy lesions.
    Input BGR, returns RGB uint8.
    """
    if img is None:
        return None
    # convert to float RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    # smaller blur
    blurred = cv2.GaussianBlur(img_rgb, (0,0), sigma)
    # milder subtraction
    enhanced = cv2.addWeighted(img_rgb, alpha, blurred, (1-alpha), 0)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    return enhanced

# ---------------------------
# CLAHE (green channel or L channel)
# ---------------------------
def apply_clahe_green(img, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Apply CLAHE to the green channel (a common practice for fundus images).
    Input: BGR or RGB uint8. We'll treat incoming as RGB to be safe.
    Returns RGB uint8.
    """
    if img is None:
        return None
    # ensure RGB
    if img.ndim != 3 or img.shape[2] != 3:
        return img
    # if BGR assume BGR -> convert to RGB
    # We'll convert BGR->RGB first (callers using our pipeline will pass BGR or RGB, we normalize)
    b,g,r = cv2.split(img[:,:,::-1]) if False else cv2.split(img)  # placeholder line, replaced below
    # simpler: detect dtype and do safe conversions:
    # We'll assume the image is in BGR (cv2 default), so transform to RGB then operate on G channel.
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.dtype == np.uint8 else img
    # operate on green channel
    g = rgb[:,:,1]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    g2 = clahe.apply(g)
    out = rgb.copy()
    out[:,:,1] = g2
    return out

def apply_clahe_lab(img, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Apply CLAHE on L channel (LAB color space) and convert back to RGB.
    Input BGR; returns RGB uint8.
    """
    if img is None:
        return None
    # ensure uint8
    if img.dtype != np.uint8:
        img_u8 = (np.clip(img,0,1)*255).astype(np.uint8)
    else:
        img_u8 = img
    # convert BGR to LAB via cv2
    lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2,a,b])
    rgb = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return rgb

# ---------------------------
# Unsharp mask (gentle)
# ---------------------------
def unsharp_mask(img, kernel_size=(5,5), sigma=1.0, amount=0.8, threshold=1):
    """
    Gentle unsharp mask. Input: RGB uint8. Output: RGB uint8.
    amount ~ 0.5-1.0 is mild.
    """
    if img is None:
        return None
    if img.dtype != np.uint8:
        img_u8 = np.clip(img*255.0, 0, 255).astype(np.uint8)
    else:
        img_u8 = img
    blurred = cv2.GaussianBlur(img_u8, kernel_size, sigma)
    diff = img_u8.astype(np.int16) - blurred.astype(np.int16)
    # threshold the difference to avoid amplifying noise
    mask = np.abs(diff) > threshold
    sharpened = img_u8.astype(np.int16) + (amount * diff).astype(np.int16)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    # preserve areas below threshold: keep original
    sharpened[~mask] = img_u8[~mask]
    return sharpened

# ---------------------------
# Normalization
# ---------------------------
def normalize_image(img):
    """
    Normalize to [0,1] float32 per-image (min-max). Input expected RGB uint8 or float.
    Returns float32 in [0,1] with shape H,W,3.
    """
    if img is None:
        return None
    if img.dtype == np.uint8:
        arr = img.astype(np.float32) / 255.0
    else:
        arr = img.astype(np.float32)
        # if already in 0..1 leave as is else scale
        if arr.max() > 1.1:
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    # tiny standardization optional - keep simple min-max for stability
    return arr.astype(np.float32)

# Backwards-compatible alias expected by debug script
def normalize(x):
    return normalize_image(x)

# ---------------------------
# Full pipeline: controlled / mild
# ---------------------------
def preprocess_retina(path, target_size=(512,512),
                      use_clahe=True, use_bengraham=False,
                      bengraham_sigma=5, bengraham_alpha=2.0,
                      use_unsharp=False):
    """
    Full, conservative preprocessing pipeline.
    Returns float32 RGB image normalized to [0,1].
    - path: path or bytes
    - target_size: (h,w)
    - use_clahe: apply CLAHE on green/L
    - use_bengraham: optional mild Ben-Graham
    - use_unsharp: optional gentle unsharp mask
    """
    try:
        logger.debug(f"Preprocessing image: {path}")
        
        img = _read_image_bgr(path)
        if img is None:
            logger.error(f"Failed to read image: {path}")
            raise PreprocessingError(f"Failed to read image: {path}", sys.exc_info())

        # Step 1: crop black borders (BGR)
        try:
            img = crop_black_borders(img, threshold=10)
            logger.debug("Step 1: Black borders cropped")
        except Exception as e:
            logger.warning(f"Black border cropping failed: {e}")

        # Step 2: detect and crop circular fundus
        try:
            img = detect_and_crop_circle(img)
            logger.debug("Step 2: Circle detected and cropped")
        except Exception as e:
            logger.warning(f"Circle detection failed: {e}")

        # Step 3: resize & pad (BGR)
        img = resize_maintain_aspect(img, target_size=target_size)
        logger.debug(f"Step 3: Resized to {target_size}")

        # Step 4: optional mild Ben Graham (returns RGB uint8)
        if use_bengraham:
            try:
                img = ben_graham_mild(img, sigma=bengraham_sigma, alpha=bengraham_alpha)
                logger.debug("Step 4: Ben Graham enhancement applied")
            except Exception as e:
                logger.warning(f"Ben Graham failed, falling back: {e}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # convert BGR -> RGB for next steps
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Step 5: CLAHE (green channel) or LAB
        if use_clahe:
            try:
                img = apply_clahe_lab(cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                      clip_limit=2.0, tile_grid_size=(8,8))
                logger.debug("Step 5: CLAHE applied (LAB)")
            except Exception as e:
                logger.warning(f"CLAHE LAB failed, trying green channel: {e}")
                img = apply_clahe_green(img, clip_limit=2.0, tile_grid_size=(8,8))

        # Step 6: optional unsharp mask (gentle)
        if use_unsharp:
            try:
                img = unsharp_mask(img, kernel_size=(5,5), sigma=1.0, amount=0.6, threshold=2)
                logger.debug("Step 6: Unsharp mask applied")
            except Exception as e:
                logger.warning(f"Unsharp mask failed: {e}")

        # Step 7: normalize to [0,1] float32
        img = normalize_image(img)
        logger.debug("Step 7: Normalized to [0,1]")

        return img
        
    except PreprocessingError:
        raise
    except Exception as e:
        logger.error(f"Preprocessing failed for {path}: {e}")
        raise PreprocessingError(f"Preprocessing failed: {e}", sys.exc_info())

# ---------------------------
# Batch helper
# ---------------------------
def preprocess_batch(paths, target_size=(512,512), use_clahe=True, **kwargs):
    """
    Preprocess list of file paths and return numpy array (N,H,W,3) float32
    kwargs forwarded to preprocess_retina.
    """
    logger.info(f"Starting batch preprocessing of {len(paths)} images")
    out = []
    success_count = 0
    error_count = 0
    
    for p in paths:
        try:
            img = preprocess_retina(p, target_size=target_size, use_clahe=use_clahe, **kwargs)
            out.append(img)
            success_count += 1
        except Exception as e:
            logger.warning(f"Preprocess failed for {p}: {e}")
            out.append(np.zeros((target_size[0], target_size[1], 3), dtype=np.float32))
            error_count += 1
    
    logger.info(f"Batch preprocessing complete: {success_count} success, {error_count} errors")
    return np.stack(out, axis=0).astype(np.float32)

# ---------------------------
# Export friendly names expected by debug script
# ---------------------------
# remove_border already defined
# resize_image already defined
# unsharp_mask defined
# apply_clahe_green defined
# normalize defined
# preprocess_retina defined
# preprocess_batch defined

# End of file
