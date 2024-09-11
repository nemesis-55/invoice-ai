# Use a base image with Python and CUDA (if needed)
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and required system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Define your working directory
WORKDIR /

# Add your file
ADD handler.py .

# Call your file when your container starts
CMD [ "python", "-u", "/handler.py" ]