# Use the RunPod PyTorch image with CUDA as the base image
# Note: This is a large image (~15-20GB). Ensure build environment has adequate disk space.
# Consider using Docker BuildKit: DOCKER_BUILDKIT=1 docker build
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set memory allocator config for CUDA to prevent memory fragmentation
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
ENV GPU_DEVICE=single
ENV NUM_GPUS=0
ENV DEEP_THINKING=false

# Set the working directory in the container
WORKDIR /

# Install system dependencies and clean up in same layer
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    tesseract-ocr \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /tmp/* /var/tmp/*

# Uninstall and reinstall torch in single layer to minimize disk usage
# This prevents having both versions on disk simultaneously
RUN pip uninstall -y torch torchvision torchaudio && \
    rm -rf /root/.cache/pip && \
    pip install --no-cache-dir \
        torch==2.6.0 \
        torchvision==0.21.0 \
        torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124 && \
    rm -rf /root/.cache/pip && \
    find /usr/local/lib/python3.11/dist-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Copy application files
COPY requirements.txt ./
COPY handler.py .
COPY gpu_config.json ./
COPY helper/ ./helper/
COPY models/ ./models/

# Install dependencies and aggressively clean up to minimize layer size
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip && \
    find /usr/local/lib/python3.11/dist-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/dist-packages -type f -name "*.pyc" -delete 2>/dev/null || true && \
    rm -rf /tmp/* /var/tmp/*

# Try to install flash_attn if available, otherwise skip (uses SDPA fallback)
RUN pip install --no-cache-dir flash_attn>=2.5.0 2>/dev/null || \
    echo "Flash attention not available, using SDPA fallback (attn_implementation=sdpa)"

# Expose port for the API
EXPOSE 8000

# Run the application
CMD ["python", "handler.py"]
