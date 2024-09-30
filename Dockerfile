FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV MODEL_NAME="Zorro123444/invoice_extracter_xylem2.1.1"
ENV MODEL_PATH="/workspace/model"
ENV GIT_LFS_SKIP_SMUDGE=1  
# Install Python and required system dependencies
RUN apt-get update && apt-get install -y \
  python3.8 \
  python3-pip \
  git \
  wget \
  curl \
  git-lfs \
  && rm -rf /var/lib/apt/lists/*

# Set Python3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a directory for storing model files
RUN mkdir -p ${MODEL_PATH}

# Clone the model repository without downloading large files
RUN git lfs install && \
    git clone https://huggingface.co/${MODEL_NAME} ${MODEL_PATH}

# Now, fetch the large files separately
RUN cd ${MODEL_PATH} && git lfs pull

# Add your handler file
COPY handler.py /workspace/handler.py

# Call your file when your container starts
CMD [ "python", "-u", "/workspace/handler.py" ]
