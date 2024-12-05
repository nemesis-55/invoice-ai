# Use a base image with CUDA and cuDNN support
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies, Python 3.9, and pip
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-distutils \
    python3-pip \
    git \
    wget \
    curl \
    tesseract-ocr \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set Python 3.9 as the default version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Upgrade pip to the latest version
RUN python -m pip install --upgrade pip

# Install PyTorch and torchvision with CUDA support
RUN pip install --no-cache-dir torch==2.1.2+cu117 torchvision==0.16.2+cu117 --index-url https://download.pytorch.org/whl/cu117

# Copy requirements.txt and install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the handler script to the container
COPY handler.py ./

# Set the entry point for the container to run your Python script
CMD ["python", "-u", "./handler.py"]
