# Use a base image with Python and CUDA
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV MODEL_NAME="Zorro123444/invoice_extracter_xylem2.1.0"
ENV MODEL_PATH="/workspace/model"

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

# Create a directory for storing model files
RUN mkdir -p ${MODEL_PATH}

# Clone the Hugging Face repository to the model path (note: no Dockerfile in HF repo)
RUN git lfs install && \
    git clone https://huggingface.co/Zorro123444/invoice_extracter_xylem2.1.0 ${MODEL_PATH}

# Define your working directory
WORKDIR /workspace

# Add your handler file
ADD handler.py .

# Define the command to run your handler
CMD [ "python", "-u", "/workspace/handler.py" ]
