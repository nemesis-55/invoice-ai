# Use the RunPod PyTorch image with CUDA as the base image
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (needed for some libraries like pytesseract, fitz)
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
RUN pip install --no-cache-dir -r requirements.txt

# Copy the handler.py file into the container
COPY handler.py .

# Expose port for the API (if you run a local server, typically 8000)
EXPOSE 8000

# Run the application
CMD ["python", "handler.py"]
