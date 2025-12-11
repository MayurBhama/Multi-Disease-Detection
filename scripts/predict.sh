#!/bin/bash

# Prediction script for DenseNet201 Diabetic Retinopathy Detection

echo "=================================="
echo "Starting DenseNet201 Prediction"
echo "=================================="

# Activate virtual environment if needed
# source venv/bin/activate

# Run prediction with TTA
python -m src.inference.predict_densenet

echo "=================================="
echo "Prediction Complete!"
echo "=================================="