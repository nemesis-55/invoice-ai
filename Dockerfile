# Base image with CUDA and cuDNN support
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, Git, and required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3-pip git wget tesseract-ocr \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set Python3 as the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Pre-install essential Python packages to avoid build issues
RUN pip install --no-cache-dir packaging setuptools wheel

# Install PyTorch explicitly
RUN pip install torch==2.0.1+cu117 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu117

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the handler script
COPY handler.py .

# Set the entry point
CMD ["python", "-u", "handler.py"]
