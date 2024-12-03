# Base image with CUDA and cuDNN support
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"

# Install Python, Git, and required system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    git \
    wget \
    tesseract-ocr \
    curl \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set Python3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Install specific Python libraries
RUN pip install \
    torch==2.1.2 \
    torchvision==0.16.2 \
    transformers==4.40.0 \
    pillow==10.1.0 \
    runpod \
    accelerate==0.30.1 \
    pymupdf \
    deepspeed \
    peft \
    timm==0.9.10 \
    sentencepiece==0.1.99 \
    tensorboardX \
    pytesseract

# Install flash_attn separately (requires CUDA toolkit)
RUN pip install flash_attn==2.3.4

# Copy the requirements.txt for reference (optional)
COPY requirements.txt .

# Copy the handler script to the container
COPY handler.py ./

# Call your file when the container starts
CMD ["python", "-u", "./handler.py"]
