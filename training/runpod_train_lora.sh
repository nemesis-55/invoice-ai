#!/bin/bash

##############################################################################
# RunPod Training Launcher - LoRA Fine-tuning
# 
# This script is optimized for RunPod GPU pods with network volume storage.
# It handles checkpoint management, automatic resumption, and proper logging.
##############################################################################

set -e  # Exit on error

echo "=========================================="
echo "RunPod Training Launcher - LoRA"
echo "=========================================="
echo "Starting at: $(date)"
echo ""

# Configuration
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
DATA_DIR="${WORKSPACE_DIR}/data"
REPO_DIR="${WORKSPACE_DIR}/invoice-ai"
OUTPUT_DIR="${DATA_DIR}/output/minicpm_v45_lora_runpod"
LOG_FILE="${OUTPUT_DIR}/training_$(date +%Y%m%d_%H%M%S).log"

# Check if we're on RunPod
if [ -d "$WORKSPACE_DIR" ]; then
    echo "✓ Detected RunPod environment"
else
    echo "⚠ Warning: Not on RunPod. Using local paths."
    WORKSPACE_DIR="."
    DATA_DIR="./data"
    REPO_DIR="."
    OUTPUT_DIR="./output/minicpm_v45_lora_runpod"
fi

# Create necessary directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$DATA_DIR"

# Start logging
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

echo "Configuration:"
echo "  Workspace: $WORKSPACE_DIR"
echo "  Data Directory: $DATA_DIR"
echo "  Repository: $REPO_DIR"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Log File: $LOG_FILE"
echo ""

# System Information
echo "=========================================="
echo "System Information"
echo "=========================================="
echo "Hostname: $(hostname)"
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""
echo "CUDA Version: $(nvcc --version | grep release || echo 'N/A')"
echo "Python Version: $(python --version)"
echo "PyTorch Version: $(python -c 'import torch; print(torch.__version__)')"
echo "Transformers Version: $(python -c 'import transformers; print(transformers.__version__)')"
echo ""

# Check data availability
echo "=========================================="
echo "Data Check"
echo "=========================================="

TRAIN_DATA="${DATA_DIR}/train_data.json"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ Error: Training data not found at $TRAIN_DATA"
    echo ""
    echo "Please prepare your dataset first:"
    echo "  1. Upload raw data to $DATA_DIR/"
    echo "  2. Run: python prepare_data/create_training_data.py"
    echo ""
    exit 1
fi

echo "✓ Training data found: $TRAIN_DATA"
TRAIN_SAMPLES=$(python -c "import json; print(len(json.load(open('$TRAIN_DATA'))))")
echo "  Samples: $TRAIN_SAMPLES"
echo ""
echo "Note: LLamaFactory will automatically split this into train/validation"
echo "      based on val_size parameter in config (default: 0.1 = 10% validation)"
echo ""

# Check for existing checkpoints
echo "=========================================="
echo "Checkpoint Check"
echo "=========================================="

if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR/checkpoint-* 2>/dev/null)" ]; then
    LATEST_CHECKPOINT=$(ls -td $OUTPUT_DIR/checkpoint-* | head -1)
    echo "✓ Found existing checkpoint: $LATEST_CHECKPOINT"
    echo "  Training will resume from this checkpoint"
    RESUME_ARG="--resume_from_checkpoint $LATEST_CHECKPOINT"
else
    echo "ℹ No existing checkpoints found. Starting fresh training."
    RESUME_ARG=""
fi
echo ""

# Environment variables
echo "=========================================="
echo "Environment Setup"
echo "=========================================="

# Set HuggingFace cache to network volume
export HF_HOME="${DATA_DIR}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "$HF_HOME"
echo "✓ HuggingFace cache: $HF_HOME"

# Check HF token
if [ -z "$HF_TOKEN" ]; then
    echo "⚠ Warning: HF_TOKEN not set. Public models only."
else
    echo "✓ HF_TOKEN configured"
fi

# Weights & Biases (optional)
if [ -n "$WANDB_API_KEY" ]; then
    echo "✓ Weights & Biases logging enabled"
    export WANDB_PROJECT="${WANDB_PROJECT:-invoice-ai-runpod}"
    echo "  Project: $WANDB_PROJECT"
fi
echo ""

# Training Configuration
echo "=========================================="
echo "Training Configuration"
echo "=========================================="

# Use LLamaFactory if available, otherwise use native scripts
if command -v llamafactory-cli &> /dev/null; then
    TRAINING_METHOD="llamafactory"
    echo "✓ Using LLamaFactory for training"
    
    # Check if config exists
    CONFIG_FILE="${REPO_DIR}/llamafactory_configs/minicpm_v45_lora.yaml"
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "❌ Error: LLamaFactory config not found at $CONFIG_FILE"
        exit 1
    fi
    
    # Create temporary config with updated paths
    TEMP_CONFIG="${OUTPUT_DIR}/training_config.yaml"
    cat "$CONFIG_FILE" | \
        sed "s|output_dir:.*|output_dir: $OUTPUT_DIR|" | \
        sed "s|dataset_dir:.*|dataset_dir: $DATA_DIR|" | \
        sed "s|data_path:.*|data_path: $TRAIN_DATA|" > "$TEMP_CONFIG"
    
    echo "  Config: $TEMP_CONFIG"
    
else
    TRAINING_METHOD="native"
    echo "✓ Using native training scripts"
    echo "  Script: ${REPO_DIR}/training/finetune_lora.sh"
fi
echo ""

# Display GPU status before training
echo "=========================================="
echo "GPU Status (Before Training)"
echo "=========================================="
nvidia-smi
echo ""

# Training command
echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo "Started at: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "Resume: ${RESUME_ARG:-No}"
echo ""
echo "Press Ctrl+C to stop training gracefully"
echo "=========================================="
echo ""

cd "$REPO_DIR"

if [ "$TRAINING_METHOD" = "llamafactory" ]; then
    # LLamaFactory training
    llamafactory-cli train "$TEMP_CONFIG" $RESUME_ARG
else
    # Native training - need to update script paths
    export DATA_PATH="$TRAIN_DATA"
    export OUTPUT_DIR="$OUTPUT_DIR"
    
    # Run training
    bash training/finetune_lora.sh $RESUME_ARG
fi

TRAINING_EXIT_CODE=$?

# Training completed
echo ""
echo "=========================================="
echo "Training Completed"
echo "=========================================="
echo "Finished at: $(date)"
echo "Exit code: $TRAINING_EXIT_CODE"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
else
    echo "❌ Training failed with exit code $TRAINING_EXIT_CODE"
fi
echo ""

# Display GPU status after training
echo "=========================================="
echo "GPU Status (After Training)"
echo "=========================================="
nvidia-smi
echo ""

# Checkpoint summary
echo "=========================================="
echo "Checkpoint Summary"
echo "=========================================="
if [ -d "$OUTPUT_DIR" ]; then
    echo "Saved checkpoints:"
    ls -lh "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | tail -5 || echo "  No checkpoints found"
    echo ""
    echo "Total output size:"
    du -sh "$OUTPUT_DIR"
else
    echo "No output directory found"
fi
echo ""

# Backup recommendation
echo "=========================================="
echo "Important: Backup Your Model!"
echo "=========================================="
echo "Your trained model is saved to:"
echo "  $OUTPUT_DIR"
echo ""
echo "Recommended: Backup to cloud storage"
echo ""
echo "Example commands:"
echo ""
echo "# Compress checkpoint"
echo "tar -czf model_checkpoint.tar.gz $OUTPUT_DIR/checkpoint-*"
echo ""
echo "# Upload to Azure Blob Storage"
echo "az storage blob upload --account-name YOUR_ACCOUNT \\"
echo "  --container-name models \\"
echo "  --file model_checkpoint.tar.gz \\"
echo "  --name model_checkpoint_$(date +%Y%m%d).tar.gz"
echo ""
echo "# Upload to AWS S3"
echo "aws s3 cp model_checkpoint.tar.gz s3://your-bucket/models/"
echo ""
echo "# Upload to Google Cloud Storage"
echo "gsutil cp model_checkpoint.tar.gz gs://your-bucket/models/"
echo ""
echo "=========================================="
echo "Log file saved to: $LOG_FILE"
echo "=========================================="

exit $TRAINING_EXIT_CODE
