# Use the RunPod PyTorch image with CUDA as the base image
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set memory allocator config for CUDA to prevent memory fragmentation
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
ENV GPU_DEVICE=single
ENV NUM_GPUS=0
ENV DEEP_THINKING=false

# Set the working directory in the container
WORKDIR /

# STEP 1: Uninstall ALL pre-installed torch components for a clean slate.
RUN pip uninstall -y torch torchvision torchaudio

# STEP 2: FIRST, install ONLY torch and its direct companions.
# This ensures torch is present before anything else tries to use it.
# Updated to torch 2.6.0 to patch torch.load RCE vulnerability
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender-dev \
    tesseract-ocr \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./



# Copy the handler.py file into the container
COPY handler.py .
COPY helper/ ./helper/
COPY models/ ./models/

RUN pip install --no-cache-dir -r requirements.txt


# Expose port for the API
EXPOSE 8000

# Run the application
CMD ["python", "handler.py"]
