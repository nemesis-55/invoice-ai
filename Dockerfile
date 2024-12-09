# Use the RunPod PyTorch image with CUDA as the base image
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /

# Install system dependencies and Git LFS in one step to reduce layers
RUN apt-get update && apt-get install -y \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender-dev \
    tesseract-ocr \
    libmagic1 \
    git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Clone repositories with authentication in a single step (avoiding temporary scripts)
RUN echo "https://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@huggingface.co" > ~/.git-credentials && \
    git config --global credential.helper store && \
    git clone https://huggingface.co/openbmb/MiniCPM-V-2_6 /models/MiniCPM-V-2_6 && \
    cd /models/MiniCPM-V-2_6 && git pull && \
    git clone https://huggingface.co/Zorro123444/invoice_extracter_2 /adaptors/invoice_extracter_2 && \
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
