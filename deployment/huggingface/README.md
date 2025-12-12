---
title: Multi-Disease Detection API
emoji: 🏥
colorFrom: blue
colorTo: cyan
sdk: docker
pinned: false
license: mit
---

# Multi-Disease Detection API

Production API for medical image classification supporting:
- **Brain MRI**: Tumor classification (glioma, meningioma, no tumor, pituitary)
- **Chest X-Ray**: Pneumonia detection
- **Retina**: Diabetic retinopathy severity (EfficientNet ensemble)

## Endpoints

- `GET /health` - Health check
- `POST /predict` - Image prediction
- `POST /gradcam` - Grad-CAM visualization
- `GET /docs` - Swagger UI

## Usage

```python
import requests

response = requests.post(
    "https://YOUR-SPACE.hf.space/predict",
    files={"file": open("brain_scan.jpg", "rb")},
    data={"disease_type": "brain_mri"}
)
print(response.json())
```
