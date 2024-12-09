# Use the RunPod PyTorch image with CUDA as the base image
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender-dev \
    tesseract-ocr \
    libmagic1 \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Install Git LFS
RUN git lfs install

# Create the authentication script for Git
RUN echo '#!/bin/bash' > /tmp/hf_token.sh && \
    echo 'echo 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' >> /tmp/hf_token.sh && \
    chmod +x /tmp/hf_token.sh

# Clone repositories using GIT_ASKPASS
RUN GIT_ASKPASS=/tmp/hf_token.sh git clone https://huggingface.co/openbmb/MiniCPM-V-2_6 /models/MiniCPM-V-2_6 && \
    cd /models/MiniCPM-V-2_6 && git pull && \
    GIT_ASKPASS=/tmp/hf_token.sh git clone https://huggingface.co/Zorro123444/invoice_extracter_2 /adaptors/invoice_extracter_2 && \
    cd /adaptors/invoice_extracter_2 && git pull

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the handler.py file into the container
COPY handler.py .

# Expose port for the API
EXPOSE 8000

# Run the application
CMD ["python", "handler.py"]
