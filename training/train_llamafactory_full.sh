#!/bin/bash

# LLamaFactory Training Script for MiniCPM-V-4.5 (Full Fine-tuning)
# This script uses LLamaFactory for full model fine-tuning with DeepSpeed

set -e

# Configuration
CONFIG_FILE="llamafactory_configs/minicpm_v45_full.yaml"
DATASET_INFO="llamafactory_configs/dataset_info.json"

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
LLAMAFACTORY_DATA_DIR=$(python -c "import llamafactory; import os; print(os.path.join(os.path.dirname(llamafactory.__file__), 'data'))" 2>/dev/null || echo "")

if [ -n "$LLAMAFACTORY_DATA_DIR" ] && [ -d "$LLAMAFACTORY_DATA_DIR" ]; then
    echo "Copying dataset_info.json to LLamaFactory data directory..."
    cp "$DATASET_INFO" "$LLAMAFACTORY_DATA_DIR/dataset_info.json"
    echo "Dataset info copied successfully!"
fi

# Print configuration
echo "=========================================="
echo "LLamaFactory Training - MiniCPM-V-4.5 (Full)"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "Dataset info: $DATASET_INFO"
echo ""
echo "Note: This requires significant GPU memory!"
echo "Recommended: 2+ GPUs with 40GB+ VRAM each"
echo ""

# Run training with LLamaFactory
echo "Starting training..."
llamafactory-cli train "$CONFIG_FILE"

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo "Model saved to: output/minicpm_v45_full_invoice"
echo "Logs available at: output/minicpm_v45_full_invoice/logs"
