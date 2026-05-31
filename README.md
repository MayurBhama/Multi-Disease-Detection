# Multi-Disease Detection System

AI-powered medical image analysis system utilizing deep learning to diagnose Brain MRIs and Chest X-Rays. The system provides clinical-grade explainability using Grad-CAM visualizations.

**Live Demo:** Hosted on Streamlit. ( https://multi-disease-detectiongit-vc6m2gtxwyzqvfryrco578.streamlit.app/ )  
**GitHub Repository:** [MayurBhama/Multi-Disease-Detection](https://github.com/MayurBhama/Multi-Disease-Detection)

---

## Features
- **Multi-Disease Analysis:** Handles distinct medical imaging modalities (Brain MRI & Chest X-Ray).
- **FastAPI Backend:** High-performance RESTful API for model inference.
- **Streamlit Frontend:** Clean, professional dashboard for clinicians.
- **Explainable AI:** Generates Grad-CAM heatmaps to visually explain model predictions.
- **Containerized:** Docker-ready for reproducible deployments.

## Models & Results

The system employs specialized Convolutional Neural Networks (CNNs) for each diagnostic pathway.

| Disease | Modality | Model Architecture | Classes | Training Data | Test Accuracy |
|---------|----------|--------------------|---------|---------------|---------------|
| **Brain Tumor** | MRI | EfficientNetB0 | Glioma, Meningioma, Pituitary, No Tumor | ~5,712 images | **97.25%** |
| **Pneumonia** | Chest X-Ray | Xception | Normal, Pneumonia | 5,216 images | **96.18%** |

## Tech Stack

- **Deep Learning:** Python, TensorFlow, Keras
- **Backend:** FastAPI, Uvicorn
- **Frontend:** Streamlit
- **Explainability:** Grad-CAM
- **Deployment:** Docker

## Project Structure

```text
configs/            model configs
scripts/            dataset preparation
src/
  api/              FastAPI backend
  meta_classifier/  core prediction engine
  preprocessing/    image preprocessing
  training/         training scripts
web/
  app.py            Streamlit UI
  api_client.py     API bridge
  styles.py         custom CSS
```

## Installation

```bash
git clone https://github.com/MayurBhama/Multi-Disease-Detection.git
cd Multi-Disease-Detection
pip install -r requirements.txt
```

## Usage

Start the backend server:
```bash
uvicorn src.api.main:app --port 8001
```

Start the frontend dashboard (in a separate terminal):
```bash
streamlit run web/app.py
```

## License
See the `LICENSE` file for details.
