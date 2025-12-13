# Dockerfile for Hugging Face Spaces
# SDK: Docker | Hardware: CPU Basic (FREE)
FROM python:3.10-slim

# Environment variables for TensorFlow
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV CUDA_VISIBLE_DEVICES=-1
ENV TF_XLA_FLAGS=--tf_xla_enable_xla_devices=false

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies (use tensorflow-cpu for smaller image)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/

# Create output directories
RUN mkdir -p outputs/gradcam static

# Hugging Face Spaces uses port 7860 by default
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
    CMD curl -f http://localhost:7860/health || exit 1

# Run FastAPI on HF Spaces default port
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
