FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV GIT_LFS_SKIP_SMUDGE=1  

# Install Python and required system dependencies
RUN apt-get update && apt-get install -y \
  python3.8 \
  python3-pip \
  git \
  wget \
  curl \
  git-lfs \
  && git lfs install \
  && rm -rf /var/lib/apt/lists/*

# Set Python3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the handler script
COPY handler.py ./

# Set model directory as an environment variable (optional)
ENV MODEL="Zorro123444/invoice_extracter"

# Call your file when the container starts
CMD ["python", "-u", "./handler.py"]