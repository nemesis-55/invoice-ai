# Use the RunPod PyTorch image with CUDA as the base image
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set memory allocator config for CUDA to prevent memory fragmentation
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
ENV GPU_DEVICE=single
ENV NUM_GPUS=0
ENV DEEP_THINKING=false

# Set the working directory in the container
WORKDIR /

# Install system dependencies first (smallest layer)
RUN apt-get update && apt-get install -y \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender-dev \
    tesseract-ocr \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# STEP 1: Uninstall and reinstall torch in a single layer to save space
# This prevents having both versions on disk simultaneously
RUN pip uninstall -y torch torchvision torchaudio && \
    pip cache purge && \
    pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 && \
    find /usr/local/lib/python3.11/dist-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Install Python dependencies and clean up in same layer to save space
COPY requirements.txt ./
COPY handler.py .
COPY helper/ ./helper/
COPY models/ ./models/

RUN pip install --no-cache-dir -r requirements.txt && \
    pip cache purge && \
    find /usr/local/lib/python3.11/dist-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/dist-packages -type f -name "*.pyc" -delete 2>/dev/null || true


# Expose port for the API
EXPOSE 8000

# Run the application
CMD ["python", "handler.py"]
