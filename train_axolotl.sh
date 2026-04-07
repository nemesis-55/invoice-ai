#!/bin/bash

# Axolotl Training Script for MiniCPM Invoice AI
# Optimized for dual RTX 5090 GPUs

set -e  # Exit on any error

echo "========================================="
echo "MiniCPM Invoice AI - Axolotl Training"
echo "Dual RTX 5090 Optimization"
echo "========================================="

# Configuration
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false

# Training configuration
CONFIG_FILE="./minicpm_axolotl_config.yaml"
DATA_DIR="./data"
OUTPUT_DIR="./output/axolotl_minicpm_lora"
DATASET_PATH="./data/train_data_axolotl.json"

# Hardware optimization for RTX 5090
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0

# Check if running in the correct directory
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE not found!"
    echo "Please ensure you're running this script from the repository root."
    exit 1
fi

# Function to check GPU availability
check_gpus() {
    echo "Checking GPU availability..."
    nvidia-smi
    
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "Found $GPU_COUNT GPU(s)"
    
    if [ $GPU_COUNT -lt 2 ]; then
        echo "Warning: Less than 2 GPUs detected. Training will proceed but may not be optimal."
        echo "This script is optimized for dual RTX 5090 setup."
    fi
}

# Function to install/update Axolotl
install_axolotl() {
    echo "Installing/Updating Axolotl and dependencies..."
    
    # Check if Axolotl is installed
    if ! python -c "import axolotl" 2>/dev/null; then
        echo "Axolotl not found. Installing..."
        
        # Install Axolotl from source for latest features
        if [ ! -d "axolotl_repo" ]; then
            git clone https://github.com/OpenAccess-AI-Collective/axolotl.git axolotl_repo
        fi
        
        cd axolotl_repo
        git pull origin main
        pip install packaging ninja
        pip install -e '.[flash-attn,deepspeed]'
        cd ..
    else
        echo "Axolotl already installed, checking version..."
        python -c "import axolotl; print(f'Axolotl version: {axolotl.__version__}')"
    fi
    
    # Install additional dependencies
    pip install --upgrade accelerate
    pip install --upgrade transformers
    pip install --upgrade datasets
    pip install --upgrade wandb
    pip install --upgrade tensorboard
    
    echo "Dependencies installation completed."
}

# Function to prepare data
prepare_data() {
    echo "Preparing dataset for Axolotl training..."
    
    # Check if original data exists
    if [ ! -f "$DATA_DIR/train_data.json" ]; then
        echo "Error: Original training data not found at $DATA_DIR/train_data.json"
        echo "Please ensure your training data is available."
        exit 1
    fi
    
    # Convert dataset to Axolotl format if not already done
    if [ ! -f "$DATASET_PATH" ]; then
        echo "Converting dataset to Axolotl format..."
        python convert_dataset_axolotl.py \
            --input "$DATA_DIR/train_data.json" \
            --output "$DATASET_PATH" \
            --image_base_path "$DATA_DIR/images" \
            --split_ratio 0.9
        
        echo "Dataset conversion completed."
    else
        echo "Axolotl format dataset already exists at $DATASET_PATH"
    fi
}

# Function to setup accelerate configuration
setup_accelerate() {
    echo "Setting up Accelerate configuration for dual GPU..."
    
    # Create accelerate config for dual GPU setup
    cat > accelerate_config.yaml << EOF
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_config_file: ./configs/deepspeed_zero2_axolotl.json
  zero3_init_flag: false
distributed_type: DEEPSPEED
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

    echo "Accelerate configuration created."
}

# Function to validate configuration
validate_config() {
    echo "Validating configuration..."
    
    # Check if config file is valid YAML
    python -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
print('Configuration file is valid YAML')
print(f'Base model: {config.get(\"base_model\", \"Not specified\")}')
print(f'Adapter type: {config.get(\"adapter\", \"Not specified\")}')
print(f'LoRA rank: {config.get(\"lora_r\", \"Not specified\")}')
"
    
    echo "Configuration validation completed."
}

# Function to run training with monitoring
run_training() {
    echo "Starting Axolotl training with dual GPU setup..."
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Start training with accelerate
    accelerate launch \
        --config_file accelerate_config.yaml \
        --main_process_port 29500 \
        -m axolotl.cli.train "$CONFIG_FILE" \
        --logging_steps 10 \
        --save_steps 1000 \
        --eval_steps 1000 \
        --warmup_steps 100 \
        --max_steps 10000
}

# Function to monitor training
monitor_training() {
    echo "Setting up training monitoring..."
    
    # Start tensorboard in background if not already running
    if ! pgrep -f tensorboard > /dev/null; then
        echo "Starting TensorBoard..."
        tensorboard --logdir="$OUTPUT_DIR" --port=6006 --bind_all &
        echo "TensorBoard started at http://localhost:6006"
    fi
    
    # Display GPU monitoring
    echo "For GPU monitoring, run: watch -n 1 nvidia-smi"
}

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up..."
    # Kill tensorboard if we started it
    pkill -f tensorboard || true
}

# Set up cleanup trap
trap cleanup EXIT

# Main execution
main() {
    echo "Starting training pipeline..."
    
    # Check prerequisites
    check_gpus
    
    # Install dependencies
    install_axolotl
    
    # Prepare data
    prepare_data
    
    # Setup accelerate
    setup_accelerate
    
    # Validate configuration
    validate_config
    
    # Setup monitoring
    monitor_training
    
    # Run training
    echo "All prerequisites met. Starting training..."
    run_training
    
    echo "Training completed!"
    echo "Check output directory: $OUTPUT_DIR"
    echo "View training logs with: tensorboard --logdir=$OUTPUT_DIR"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-install)
            SKIP_INSTALL=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --config FILE         Axolotl config file (default: $CONFIG_FILE)"
            echo "  --data-dir DIR        Data directory (default: $DATA_DIR)"
            echo "  --output-dir DIR      Output directory (default: $OUTPUT_DIR)"
            echo "  --skip-install        Skip Axolotl installation"
            echo "  --dry-run            Validate setup without running training"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Handle dry run
if [ "$DRY_RUN" = true ]; then
    echo "Dry run mode - validating setup only..."
    check_gpus
    validate_config
    echo "Dry run completed successfully!"
    exit 0
fi

# Skip installation if requested
if [ "$SKIP_INSTALL" = true ]; then
    echo "Skipping Axolotl installation as requested..."
    prepare_data
    setup_accelerate
    validate_config
    monitor_training
    run_training
else
    main
fi