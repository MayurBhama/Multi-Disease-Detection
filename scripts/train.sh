#!/bin/bash

# Training script for DenseNet201 Diabetic Retinopathy Detection

echo "=================================="
echo "Starting DenseNet201 Training"
echo "=================================="

# Activate virtual environment if needed
# source venv/bin/activate

# Run training
python -m src.training.train_densenet

echo "=================================="
echo "Training Complete!"
echo "=================================="