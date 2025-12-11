"""
Script to preprocess and save retina images to data/processed/retina directory.
This allows for faster training by pre-computing all preprocessing steps.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.retina_preprocessing import preprocess_retina
from src.utils.logger import get_logger
from src.utils.exception import PreprocessingError

# Initialize logger
logger = get_logger(__name__)

# Configuration
TRAIN_CSV = "data/raw/retina/train.csv"
TEST_CSV = "data/raw/retina/test.csv"
TRAIN_DIR = "data/raw/retina/train_images"
TEST_DIR = "data/raw/retina/test_images"
OUTPUT_TRAIN_DIR = "data/processed/retina/train_images"
OUTPUT_TEST_DIR = "data/processed/retina/test_images"
IMAGE_SIZE = (384, 384)


def save_preprocessed_images(csv_path, input_dir, output_dir, image_size=(384, 384)):
    """
    Preprocess and save images.
    
    Args:
        csv_path: Path to CSV file with image IDs
        input_dir: Input directory with raw images
        output_dir: Output directory for preprocessed images
        image_size: Target image size (height, width)
    """
    logger.info(f"Starting preprocessing from {csv_path}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CSV
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} images from CSV")
    except Exception as e:
        raise PreprocessingError(f"Failed to load CSV: {csv_path}", sys.exc_info())
    
    # Process images
    success_count = 0
    error_count = 0
    errors = []
    
    logger.info("Starting image preprocessing...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        img_name = f"{row['id_code']}.png"
        input_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, img_name)
        
        # Skip if already processed
        if os.path.exists(output_path):
            logger.debug(f"Skipping {img_name} - already processed")
            success_count += 1
            continue
        
        try:
            # Preprocess image
            img = preprocess_retina(
                input_path,
                target_size=image_size,
                use_clahe=True,
                use_bengraham=True,
                use_unsharp=False
            )
            
            # Convert back to uint8 for storage
            img_uint8 = (img * 255).astype(np.uint8)
            
            # Convert RGB to BGR for cv2.imwrite
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            
            # Save processed image
            success = cv2.imwrite(output_path, img_bgr)
            
            if success:
                success_count += 1
            else:
                error_count += 1
                errors.append(f"{img_name}: Failed to write image")
                logger.warning(f"Failed to write: {img_name}")
                
        except Exception as e:
            error_count += 1
            error_msg = f"{img_name}: {str(e)}"
            errors.append(error_msg)
            logger.error(f"Error processing {img_name}: {str(e)}")
    
    # Summary
    logger.info("="*60)
    logger.info("Preprocessing Summary")
    logger.info("="*60)
    logger.info(f"Successfully processed: {success_count}/{len(df)}")
    logger.info(f"Errors: {error_count}")
    
    if errors:
        logger.warning(f"First 5 errors:")
        for error in errors[:5]:
            logger.warning(f"  - {error}")
    
    logger.info(f"Preprocessed images saved to: {output_dir}")
    
    return success_count, error_count


def main():
    """Main execution"""
    logger.info("="*60)
    logger.info("Retina Image Preprocessing Pipeline")
    logger.info("="*60)
    
    # Process training images
    logger.info("\n1. Processing training images...")
    train_success, train_errors = save_preprocessed_images(
        TRAIN_CSV,
        TRAIN_DIR,
        OUTPUT_TRAIN_DIR,
        IMAGE_SIZE
    )
    
    # Process test images
    logger.info("\n2. Processing test images...")
    test_success, test_errors = save_preprocessed_images(
        TEST_CSV,
        TEST_DIR,
        OUTPUT_TEST_DIR,
        IMAGE_SIZE
    )
    
    # Overall summary
    logger.info("\n" + "="*60)
    logger.info("OVERALL SUMMARY")
    logger.info("="*60)
    logger.info(f"Training images: {train_success} success, {train_errors} errors")
    logger.info(f"Test images: {test_success} success, {test_errors} errors")
    logger.info(f"Total processed: {train_success + test_success}")
    logger.info("="*60)
    
    logger.info("\n✓ Preprocessing pipeline complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Pipeline failed: {str(e)}", exc_info=True)
        raise
