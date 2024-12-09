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
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./



# Copy the handler.py file into the container
COPY handler.py .

RUN pip install --no-cache-dir -r requirements.txt


# Expose port for the API
EXPOSE 8000

# Run the application
CMD ["python", "handler.py"]
