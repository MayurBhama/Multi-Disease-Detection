"""
Kaggle Submission Generator
===========================
Generate predictions on test images using trained EfficientNet ensemble.
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import logging and exception handling
from src.utils.logger import get_logger
from src.utils.exception import ModelLoadError, PredictionError, PreprocessingError

# Initialize logger
logger = get_logger(__name__)

# =====================================================
# CONFIGURATION
# =====================================================
IMAGE_SIZE = (384, 384)
TEST_CSV = "data/raw/retina/test.csv"
TEST_DIR = "data/raw/retina/test_images"
OUTPUT_DIR = "outputs/production"
MODEL_DIR = "outputs/production/models"

# =====================================================
# PREPROCESSING (same as training)
# =====================================================
def preprocess_image(image_path, target_size=IMAGE_SIZE):
    """Production preprocessing."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Crop black borders
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        margin = 5
        x, y = max(0, x - margin), max(0, y - margin)
        w = min(img.shape[1] - x, w + 2 * margin)
        h = min(img.shape[0] - y, h + 2 * margin)
        img = img[y:y+h, x:x+w]
    
    # Resize
    h, w = img.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Pad
    pad_h = (target_size[0] - new_h) // 2
    pad_w = (target_size[1] - new_w) // 2
    img = cv2.copyMakeBorder(
        img, pad_h, target_size[0] - new_h - pad_h,
        pad_w, target_size[1] - new_w - pad_w,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    
    # EfficientNet preprocessing
    img = img.astype(np.float32)
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    
    return img


# =====================================================
# MODEL BUILDING
# =====================================================
def build_model(variant='s'):
    """Build model architecture."""
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name='input')
    
    if variant == 's':
        base = tf.keras.applications.EfficientNetV2S(
            include_top=False, weights=None, input_tensor=inputs, pooling='avg'
        )
    elif variant == 'b0':
        base = tf.keras.applications.EfficientNetV2B0(
            include_top=False, weights=None, input_tensor=inputs, pooling='avg'
        )
    elif variant == 'b2':
        base = tf.keras.applications.EfficientNetV2B2(
            include_top=False, weights=None, input_tensor=inputs, pooling='avg'
        )
    
    x = base.output
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs, name=f'efficientnet_{variant}')


# =====================================================
# TEST-TIME AUGMENTATION
# =====================================================
def predict_with_tta(model, img, num_augments=5):
    """Predict with TTA."""
    augments = [
        img,
        np.fliplr(img),
        np.flipud(img),
        np.rot90(img, k=1),
        np.rot90(img, k=3)
    ][:num_augments]
    
    batch = np.stack(augments)
    preds = model.predict(batch, verbose=0)
    return np.mean(preds, axis=0)


# =====================================================
# MAIN
# =====================================================
def generate_submission():
    """Generate submission file with predictions."""
    logger.info("="*60)
    logger.info("KAGGLE SUBMISSION GENERATOR")
    logger.info("EfficientNet Ensemble + TTA")
    logger.info("="*60)
    
    try:
        # Load test data
        logger.info(f"Loading test data from {TEST_CSV}")
        test_df = pd.read_csv(TEST_CSV)
        logger.info(f"Test images: {len(test_df)}")
        
        # Load models
        logger.info("Loading models...")
        models = []
        variants = ['s', 'b0', 'b2']
        weights = [0.336, 0.326, 0.338]  # From training
        
        for var in variants:
            try:
                logger.info(f"Loading EfficientNetV2-{var.upper()}...")
                model = build_model(var)
                weight_path = os.path.join(MODEL_DIR, f'efficientnet_{var}_best.h5')
                
                if os.path.exists(weight_path):
                    model.load_weights(weight_path)
                    logger.info(f"✓ Loaded from {weight_path}")
                else:
                    # Try final weights
                    weight_path = os.path.join(MODEL_DIR, f'efficientnet_{var}_final.h5')
                    if os.path.exists(weight_path):
                        model.load_weights(weight_path)
                        logger.info(f"✓ Loaded from {weight_path}")
                    else:
                        logger.error(f"✗ Weights not found for {var}!")
                        raise ModelLoadError(f"Weights not found: {weight_path}", sys.exc_info())
                
                models.append(model)
            except ModelLoadError:
                raise
            except Exception as e:
                logger.error(f"Failed to load model {var}: {e}")
                raise ModelLoadError(f"Failed to load model {var}: {e}", sys.exc_info())
        
        # Generate predictions
        logger.info("Generating predictions with TTA...")
        predictions = []
        success_count = 0
        error_count = 0
        
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
            img_path = os.path.join(TEST_DIR, f"{row['id_code']}.png")
            
            try:
                img = preprocess_image(img_path)
                
                # Ensemble prediction with TTA
                ensemble_pred = np.zeros(5)
                for model, weight in zip(models, weights):
                    pred = predict_with_tta(model, img)
                    ensemble_pred += pred * weight
                
                # Get class with highest probability
                diagnosis = np.argmax(ensemble_pred)
                predictions.append({
                    'id_code': row['id_code'],
                    'diagnosis': int(diagnosis)
                })
                success_count += 1
            except Exception as e:
                logger.warning(f"Error on {row['id_code']}: {e}")
                predictions.append({
                    'id_code': row['id_code'],
                    'diagnosis': 0  # Default to No DR
                })
                error_count += 1
        
        logger.info(f"Predictions complete: {success_count} success, {error_count} errors")
        
        # Create submission DataFrame
        submission_df = pd.DataFrame(predictions)
        
        # Save submission
        submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
        submission_df.to_csv(submission_path, index=False)
        
        logger.info(f"✓ Submission saved to: {submission_path}")
        
        # Show distribution
        logger.info("Prediction distribution:")
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
        for cls in range(5):
            count = (submission_df['diagnosis'] == cls).sum()
            pct = 100 * count / len(submission_df)
            logger.info(f"  {class_names[cls]:15}: {count:4} ({pct:.1f}%)")
        
        logger.info("="*60)
        logger.info("SUBMISSION COMPLETE!")
        logger.info("="*60)
        
        return submission_df
        
    except ModelLoadError:
        raise
    except Exception as e:
        logger.critical(f"Submission generation failed: {e}")
        raise PredictionError(f"Submission generation failed: {e}", sys.exc_info())


if __name__ == "__main__":
    try:
        submission_df = generate_submission()
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)
