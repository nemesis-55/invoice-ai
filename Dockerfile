# Use a base image with CUDA and cuDNN support
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies and Python 3.10 in one step to reduce image size
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set Python 3.10 as the default version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip to the latest version
RUN python -m pip install --upgrade pip

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the handler script to the container
COPY handler.py ./

# Run the handler script when the container starts
CMD ["python", "-u", "./handler.py"]
