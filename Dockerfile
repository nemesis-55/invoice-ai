FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    git \
    wget \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set Python 3.9 as the default version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Copy requirements.txt and install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the handler script to the container
COPY handler.py ./

# Call your file when the container starts
CMD ["python", "-u", "./handler.py"]
