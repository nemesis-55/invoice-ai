# Use a base image with CUDA and cuDNN development tools
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies and Python 3.9
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-distutils \
    python3-pip \
    git \
    wget \
    curl \
    tesseract-ocr \
    build-essential \
    python3.9-dev \
    cmake \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set Python 3.9 as the default version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Upgrade pip to the latest version
RUN python -m pip install --upgrade pip

RUN pip install packaging

# Install PyTorch and torchvision with CUDA support
RUN pip install --no-cache-dir torch==2.0.1+cu117 torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117

# Install flash_attn (this now works because the build tools and CUDA development toolkit are available)
RUN pip install flash_attn

# Copy requirements.txt and install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the handler script to the container
COPY handler.py ./

# Set the entry point for the container to run your Python script
CMD ["python", "-u", "./handler.py"]
