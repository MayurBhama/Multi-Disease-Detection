import os
from pathlib import Path

project_name = "multi-med-detect"

# --- FULL DIRECTORY STRUCTURE FOR OUR PROJECT ---
list_of_files = [

    # ---------------- ROOT FILES ----------------
    "requirements.txt",

    # ---------------- DATA ----------------
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/meta/.gitkeep",
    "data/uploads/.gitkeep",

    # ---------------- MODELS ----------------
    "models/pneumonia/.gitkeep",
    "models/brain/.gitkeep",
    "models/retina/.gitkeep",
    "models/meta/.gitkeep",

    # ---------------- NOTEBOOKS ----------------
    "notebooks/eda.ipynb",
    "notebooks/pneumonia_training.ipynb",
    "notebooks/brain_training.ipynb",
    "notebooks/retina_training.ipynb",
    "notebooks/meta_classifier_training.ipynb",

    # ---------------- SRC PACKAGE ----------------
    "src/__init__.py",

    # PREPROCESSING
    "src/preprocessing/__init__.py",
    "src/preprocessing/preprocess_xray.py",
    "src/preprocessing/preprocess_mri.py",
    "src/preprocessing/preprocess_retina.py",

    # META CLASSIFIER
    "src/meta_classifier/__init__.py",
    "src/meta_classifier/train_meta.py",
    "src/meta_classifier/infer_meta.py",

    # INFERENCE PIPELINE
    "src/inference/__init__.py",
    "src/inference/model_loader.py",
    "src/inference/router.py",
    "src/inference/unified_predict.py",
    "src/inference/gradcam_tf.py",

    # TRAINING SCRIPTS (Fine-tuning)
    "src/training/__init__.py",
    "src/training/train_pneumonia.py",
    "src/training/train_brain.py",
    "src/training/train_retina.py",

    # API
    "src/api/__init__.py",
    "src/api/main.py",

    # WEB UI
    "web/__init__.py",
    "web/app.py",

    # DEPLOYMENT
    "deployment/Dockerfile",
    "deployment/docker-compose.yaml",

    # TESTS
    "tests/__init__.py",
    "tests/test_inference.py",
    "tests/test_api.py",
]


def create_structure():
    for filepath in list_of_files:
        filepath = Path(project_name) / filepath
        filedir, filename = os.path.split(filepath)

        # Create folders
        if filedir != "":
            os.makedirs(filedir, exist_ok=True)

        # Create empty placeholder files
        if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
            with open(filepath, "w") as f:
                # Autogenerate based on extension
                if filename.endswith(".py"):
                    f.write("# Auto-generated file\n")
                elif filename.endswith(".gitignore"):
                    f.write("venv/\n__pycache__/\n*.h5\n*.pth\nmodels/\ndata/\n.DS_Store\n.ipynb_checkpoints/\n")
                elif filename == "requirements.txt":
                    f.write(
                        "tensorflow>=2.10\n"
                        "numpy\n"
                        "pillow\n"
                        "opencv-python\n"
                        "fastapi\n"
                        "uvicorn\n"
                        "streamlit\n"
                        "scikit-learn\n"
                        "albumentations\n"
                        "pydantic\n"
                        "matplotlib\n"
                        "tqdm\n"
                    )
                elif filename == "README.md":
                    f.write(
                        "# Multi-Med Detect\n\n"
                        "Unified Medical Diagnosis AI System:\n"
                        "- Pneumonia Detection (X-Ray)\n"
                        "- Brain Tumor Classification (MRI)\n"
                        "- Diabetic Retinopathy Grading (Retina)\n"
                        "- Auto-routing Meta Classifier\n"
                        "- Grad-CAM Explanations\n"
                        "- FastAPI Backend + Streamlit UI\n\n"
                        "## Project Structure Auto-Generated\n"
                    )
                elif filename == "LICENSE":
                    f.write("MIT License\n")
                else:
                    f.write("")
            print(f" Created: {filepath}")
        else:
            print(f" Exists:  {filepath}")


if __name__ == "__main__":
    print(f"\nInitializing project structure for: {project_name}\n")
    create_structure()
    print(f"\nProject structure created successfully in folder: {project_name}/\n")
