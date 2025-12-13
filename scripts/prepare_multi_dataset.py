"""
Multi-Dataset Brain MRI - Data Preparation Script
==================================================
Combines multiple datasets into unified 5-class structure:
- glioma, meningioma, notumor, pituitary, other_tumor

Run: python scripts/prepare_multi_dataset.py
"""

import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_COMBINED = PROJECT_ROOT / "data" / "combined_brain"

# Source datasets
CURRENT_DATASET = DATA_RAW / "brain_mri"
BR35H_DATASET = DATA_RAW / "br35h"

# Classes for unified dataset
CLASSES = ["glioma", "meningioma", "notumor", "pituitary", "other_tumor"]

# Train/test split ratio
TEST_SPLIT = 0.2


def create_directory_structure():
    """Create unified dataset directory structure."""
    print("[1/4] Creating directory structure...")
    
    for split in ["train", "test"]:
        for cls in CLASSES:
            (DATA_COMBINED / split / cls).mkdir(parents=True, exist_ok=True)
    
    print(f"  Created: {DATA_COMBINED}")


def copy_current_dataset():
    """Copy images from current 4-class dataset."""
    print("[2/4] Copying current dataset (4 classes)...")
    
    class_mapping = {
        "glioma": "glioma",
        "meningioma": "meningioma", 
        "notumor": "notumor",
        "pituitary": "pituitary"
    }
    
    for split_src, split_dst in [("Training", "train"), ("Testing", "test")]:
        src_dir = CURRENT_DATASET / split_src
        if not src_dir.exists():
            print(f"  Warning: {src_dir} not found")
            continue
            
        for src_class, dst_class in class_mapping.items():
            src_class_dir = src_dir / src_class
            if not src_class_dir.exists():
                continue
                
            dst_class_dir = DATA_COMBINED / split_dst / dst_class
            
            images = list(src_class_dir.glob("*.jpg")) + list(src_class_dir.glob("*.jpeg")) + list(src_class_dir.glob("*.png"))
            
            for img in tqdm(images, desc=f"  {src_class} -> {dst_class}", leave=False):
                dst_path = dst_class_dir / f"orig_{img.name}"
                if not dst_path.exists():
                    shutil.copy2(img, dst_path)
    
    print("  Done")


def copy_br35h_dataset():
    """Copy Br35H dataset (yes -> other_tumor, no -> notumor)."""
    print("[3/4] Copying Br35H dataset...")
    
    if not BR35H_DATASET.exists():
        print(f"  ERROR: Br35H dataset not found at {BR35H_DATASET}")
        print("  Please download from: https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection")
        print("  And extract to: data/raw/br35h/")
        return False
    
    # Br35H has 'yes' (tumor) and 'no' (no tumor) folders
    class_mapping = {
        "yes": "other_tumor",
        "no": "notumor"
    }
    
    for src_class, dst_class in class_mapping.items():
        src_dir = BR35H_DATASET / src_class
        if not src_dir.exists():
            # Try alternative names
            alt_names = [src_class, src_class.upper(), src_class.capitalize()]
            for alt in alt_names:
                if (BR35H_DATASET / alt).exists():
                    src_dir = BR35H_DATASET / alt
                    break
            else:
                print(f"  Warning: {src_dir} not found")
                continue
        
        images = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.jpeg")) + list(src_dir.glob("*.png"))
        
        # Shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * (1 - TEST_SPLIT))
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Copy train
        dst_train = DATA_COMBINED / "train" / dst_class
        for img in tqdm(train_images, desc=f"  {src_class} -> {dst_class} (train)", leave=False):
            dst_path = dst_train / f"br35h_{img.name}"
            if not dst_path.exists():
                shutil.copy2(img, dst_path)
        
        # Copy test
        dst_test = DATA_COMBINED / "test" / dst_class
        for img in tqdm(test_images, desc=f"  {src_class} -> {dst_class} (test)", leave=False):
            dst_path = dst_test / f"br35h_{img.name}"
            if not dst_path.exists():
                shutil.copy2(img, dst_path)
    
    print("  Done")
    return True


def print_summary():
    """Print dataset statistics."""
    print("\n[4/4] Dataset Summary:")
    print("=" * 50)
    
    total_train = 0
    total_test = 0
    
    for cls in CLASSES:
        train_count = len(list((DATA_COMBINED / "train" / cls).glob("*")))
        test_count = len(list((DATA_COMBINED / "test" / cls).glob("*")))
        total_train += train_count
        total_test += test_count
        print(f"  {cls:15} | Train: {train_count:5} | Test: {test_count:4}")
    
    print("-" * 50)
    print(f"  {'TOTAL':15} | Train: {total_train:5} | Test: {total_test:4}")
    print("=" * 50)
    print(f"\nCombined dataset saved to: {DATA_COMBINED}")


def main():
    print("=" * 50)
    print("Multi-Dataset Preparation for Brain MRI")
    print("=" * 50)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Step 1: Create directories
    create_directory_structure()
    
    # Step 2: Copy current dataset
    copy_current_dataset()
    
    # Step 3: Copy Br35H dataset
    success = copy_br35h_dataset()
    
    # Step 4: Print summary
    print_summary()
    
    if not success:
        print("\n⚠️  Br35H dataset was not found. Please download it first.")
        print("   The current 4-class dataset has been prepared.")
    else:
        print("\n✓ Dataset preparation complete!")


if __name__ == "__main__":
    main()
