#!/bin/bash

# LLamaFactory Training Script for MiniCPM-V-4.5 (LoRA)
# This script uses LLamaFactory for efficient fine-tuning with LoRA

set -e

# Resolve repo root and run from there so relative paths work
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$SCRIPT_DIR"

# Configuration
DATASET_INFO="llamafactory_configs/dataset_info.json"

# Use incremental config if --incremental flag is passed
if [ "$1" = "--incremental" ]; then
    CONFIG_FILE="llamafactory_configs/minicpm_v45_incremental_lora.yaml"
    echo "Mode: INCREMENTAL TRAINING (continuing from previous adapter)"
else
    CONFIG_FILE="llamafactory_configs/minicpm_v45_lora.yaml"
    echo "Mode: INITIAL TRAINING (fresh LoRA)"
fi

# Check if LLamaFactory is installed
if ! python -c "import llamafactory" 2>/dev/null; then
    echo "Error: LLamaFactory is not installed!"
    echo "Please run: pip install llamafactory[torch,metrics,deepspeed,minicpm_v]"
    exit 1
fi

# Check if dataset_info.json exists
if [ ! -f "$DATASET_INFO" ]; then
    echo "Error: Dataset info file not found at $DATASET_INFO"
    exit 1
fi

# Copy dataset_info.json to LLamaFactory data directory
# This is required for LLamaFactory to recognize the dataset
LLAMAFACTORY_DATA_DIR=$(python -c "import llamafactory; import os; print(os.path.join(os.path.dirname(llamafactory.__file__), 'data'))" 2>/dev/null || echo "")

if [ -n "$LLAMAFACTORY_DATA_DIR" ] && [ -d "$LLAMAFACTORY_DATA_DIR" ]; then
    echo "Copying dataset_info.json to LLamaFactory data directory..."
    cp "$DATASET_INFO" "$LLAMAFACTORY_DATA_DIR/dataset_info.json"
    echo "Dataset info copied successfully!"
fi

# Print configuration
echo "=========================================="
echo "LLamaFactory Training - MiniCPM-V-4.5 (LoRA)"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "Dataset info: $DATASET_INFO"
echo ""

# Run training with LLamaFactory
echo "Starting training..."
llamafactory-cli train "$CONFIG_FILE"

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo "Config used: $CONFIG_FILE"
