# Base image with CUDA and cuDNN support
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, Git, and required system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    git \
    wget \
    tesseract-ocr \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*



# Set Python3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install Python dependencies from the requirements file
RUN pip install packaging
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip

# Copy the handler script to the container
COPY handler.py ./

# Call your file when the container starts
CMD ["python", "-u", "./handler.py"]
