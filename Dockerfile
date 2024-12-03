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
    build-essential \
    cmake \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set Python3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install pip and its dependencies
RUN pip install --upgrade pip setuptools wheel

# Preinstall required modules to avoid build failures
RUN pip install --no-cache-dir packaging==23.2

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip

# Pre-download large dependencies if needed
RUN pip install --no-cache-dir http://thunlp.oss-cn-qingdao.aliyuncs.com/multi_modal/never_delete/modelscope_studio-0.4.0.9-py3-none-any.whl

# Ensure pytesseract and decord are available
RUN pip install --no-cache-dir pytesseract decord

# Copy the handler script to the container
COPY handler.py ./

# Set the container startup command
CMD ["python", "-u", "./handler.py"]
