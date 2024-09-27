# Use a base image with Python and CUDA
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV MODEL_NAME="openbmb/MiniCPM-Llama3-V-2_5"
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

# Download the model and tokenizer at build time and store them in the specified model path
RUN python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('${MODEL_NAME}', trust_remote_code=True, cache_dir='${MODEL_PATH}'); \
    AutoModelForCausalLM.from_pretrained('${MODEL_NAME}', trust_remote_code=True, cache_dir='${MODEL_PATH}')"

# Define your working directory
WORKDIR /workspace

# Add your handler file
ADD handler.py .

# Call your file when your container starts
CMD [ "python", "-u", "/workspace/handler.py" ]
