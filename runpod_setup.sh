#!/bin/bash

##############################################################################
# RunPod Setup Script
# 
# Quick setup script for first-time RunPod pod initialization.
# Run this once when you first deploy a pod.
##############################################################################

set -e

echo "=========================================="
echo "RunPod Setup Script"
echo "=========================================="
echo "This script will set up your RunPod pod for training"
echo ""

# Detect environment
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"

if [ -d "$WORKSPACE_DIR" ]; then
    echo "✓ Detected RunPod environment"
    echo "  Workspace: $WORKSPACE_DIR"
else
    echo "⚠ Not on RunPod, using current directory"
    WORKSPACE_DIR="$(pwd)"
fi

cd "$WORKSPACE_DIR"

# Step 1: Clone repository
echo ""
echo "=========================================="
echo "Step 1: Clone Repository"
echo "=========================================="

REPO_DIR="${WORKSPACE_DIR}/invoice-ai"

if [ -d "$REPO_DIR" ]; then
    echo "Repository already exists at: $REPO_DIR"
    read -p "Pull latest changes? (Y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        cd "$REPO_DIR"
        git pull
        echo "✓ Updated to latest version"
    fi
else
    echo "Cloning repository..."
    git clone https://github.com/Arindam2002/invoice-ai.git
    echo "✓ Repository cloned to: $REPO_DIR"
fi

cd "$REPO_DIR"

# Step 2: Install dependencies
echo ""
echo "=========================================="
echo "Step 2: Install Dependencies"
echo "=========================================="

echo "Upgrading pip..."
pip install --upgrade pip

echo ""
echo "Installing requirements..."
pip install -r requirements.txt

echo ""
read -p "Install LLamaFactory? (recommended) (Y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo "Installing LLamaFactory..."
    pip install llamafactory[torch,metrics,deepspeed,minicpm_v]
    echo "✓ LLamaFactory installed"
else
    echo "⊘ Skipping LLamaFactory installation"
fi

# Step 3: Validate configuration
echo ""
echo "=========================================="
echo "Step 3: Validate Configuration"
echo "=========================================="

python validate_config.py

# Step 4: Set up data directory
echo ""
echo "=========================================="
echo "Step 4: Set Up Data Directory"
echo "=========================================="

DATA_DIR="${WORKSPACE_DIR}/data"
mkdir -p "$DATA_DIR"
echo "✓ Data directory: $DATA_DIR"

# Create subdirectories
mkdir -p "$DATA_DIR/output"
mkdir -p "$DATA_DIR/hf_cache"
mkdir -p "$DATA_DIR/checkpoints"

echo "✓ Created subdirectories:"
echo "  - $DATA_DIR/output (for training outputs)"
echo "  - $DATA_DIR/hf_cache (for HuggingFace cache)"
echo "  - $DATA_DIR/checkpoints (for manual backups)"

# Step 5: Environment setup
echo ""
echo "=========================================="
echo "Step 5: Environment Setup"
echo "=========================================="

ENV_FILE="${WORKSPACE_DIR}/.env"

if [ ! -f "$ENV_FILE" ]; then
    cat > "$ENV_FILE" << 'EOF'
# HuggingFace Token (required for model downloads)
HF_TOKEN=your_token_here

# Data paths
RAW_DATA_OUTPUT=/workspace/data/raw_data.json
TRAIN_DATA_PATH=/workspace/data/train_data.json

# Output directory
OUTPUT_DIR=/workspace/data/output

# Weights & Biases (optional)
WANDB_API_KEY=your_wandb_key
WANDB_PROJECT=invoice-ai-runpod

# HuggingFace cache
HF_HOME=/workspace/data/hf_cache
TRANSFORMERS_CACHE=/workspace/data/hf_cache
EOF
    echo "✓ Created environment file: $ENV_FILE"
    echo ""
    echo "⚠ IMPORTANT: Edit $ENV_FILE and add your tokens!"
    echo ""
    echo "Required:"
    echo "  - HF_TOKEN: Your HuggingFace token"
    echo ""
    echo "Optional:"
    echo "  - WANDB_API_KEY: For experiment tracking"
    echo ""
    echo "Note: Data splitting is handled by LLamaFactory during training (val_size parameter)"
else
    echo "✓ Environment file already exists: $ENV_FILE"
fi

echo ""
echo "To load environment variables, run:"
echo "  source $ENV_FILE"

# Step 6: System information
echo ""
echo "=========================================="
echo "Step 6: System Information"
echo "=========================================="

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Step 7: Quick start guide
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Configure your tokens:"
echo "   nano $ENV_FILE"
echo "   source $ENV_FILE"
echo ""
echo "2. Upload your training data to:"
echo "   $DATA_DIR/"
echo ""
echo "3. Prepare dataset:"
echo "   cd $REPO_DIR"
echo "   python prepare_data/create_training_data.py"
echo ""
echo "4. Start training:"
echo "   # For LoRA (recommended):"
echo "   bash training/runpod_train_lora.sh"
echo ""
echo "   # For full fine-tuning (multi-GPU):"
echo "   bash training/runpod_train_full.sh"
echo ""
echo "For detailed instructions, see:"
echo "  $REPO_DIR/RUNPOD_TRAINING_GUIDE.md"
echo ""
echo "=========================================="
