# Base image with CUDA and cuDNN support
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV GIT_LFS_SKIP_SMUDGE=1  

# Install Python, Git, and required system dependencies
RUN apt-get update && apt-get install -y \
  python3.8 \
  python3-pip \
  git \
  git-lfs \
  wget \
  curl \
  && rm -rf /var/lib/apt/lists/*

# Initialize Git LFS
RUN git lfs install

# Set Python3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install Python dependencies from the requirements file
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set model and adapter paths as environment variables (optional)
ENV MODEL_DIR="/app/model"
ENV ADAPTER_DIR="/app/adapter"

# Create directories for models and adapters
RUN mkdir -p $MODEL_DIR $ADAPTER_DIR

# Clone the model repository (using Git LFS)
RUN git clone https://huggingface.co/Zorro123444/openbmb/MiniCPM-Llama3-V-2_5 $MODEL_DIR && \
    cd $MODEL_DIR && git lfs pull

# Clone the adapter repository (using Git LFS)
RUN git clone https://huggingface.co/Zorro123444/Zorro123444/xylem_invoice_extracter $ADAPTER_DIR && \
    cd $ADAPTER_DIR && git lfs pull

# Copy the handler script to the container
COPY handler.py ./

# Call your file when the container starts
CMD ["python", "-u", "./handler.py"]
