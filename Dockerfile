# Base image with CUDA and cuDNN support
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

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

# Install specific Python libraries
RUN pip install \
    "urllib3<1.27,>=1.25.4" \
    torch \
    torchvision \
    transformers \
    pillow \
    runpod \
    accelerate \
    pymupdf \
    peft \
    timm \
    sentencepiece \
    pytesseract

# Install flash_attn separately (requires CUDA toolkit)
RUN pip install flash_attn

# Copy the handler script to the container
COPY handler.py ./

# Call your file when the container starts
CMD ["python", "-u", "./handler.py"]
