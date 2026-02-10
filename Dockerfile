# Use the RunPod PyTorch runtime image (smaller than devel)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-runtime-ubuntu22.04

# Set memory allocator config for CUDA to prevent memory fragmentation
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
ENV GPU_DEVICE=single
ENV NUM_GPUS=0
ENV DEEP_THINKING=false

# Set the working directory in the container
WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    tesseract-ocr \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Uninstall pre-installed torch and install target version
# Combined into single layer to minimize disk usage
RUN pip uninstall -y torch torchvision torchaudio && \
    pip cache purge && \
    pip install --no-cache-dir \
        torch==2.6.0 \
        torchvision==0.21.0 \
        torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124

# Copy application files
COPY requirements.txt ./
COPY handler.py .
COPY helper/ ./helper/
COPY models/ ./models/

# Install Python dependencies (core requirements without flash_attn)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Try to install flash_attn if pre-built wheel is available, otherwise skip
# The model uses attn_implementation="sdpa" which works without flash_attn
RUN pip install --no-cache-dir flash_attn>=2.5.0 || \
    echo "Flash attention not available, using SDPA fallback"

# Expose port for the API
EXPOSE 8000

# Run the application
CMD ["python", "handler.py"]
