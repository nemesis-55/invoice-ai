#!/bin/bash

# Setup script for Axolotl migration
# This script prepares the environment for migrating to Axolotl

echo "========================================="
echo "Invoice AI - Axolotl Migration Setup"
echo "========================================="

# Function to check Python version
check_python() {
    echo "Checking Python version..."
    python_version=$(python --version 2>&1 | awk '{print $2}')
    required_version="3.8"
    
    if python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        echo "✓ Python $python_version is compatible"
    else
        echo "✗ Python $python_version is not compatible. Please install Python 3.8 or higher."
        exit 1
    fi
}

# Function to check CUDA availability
check_cuda() {
    echo "Checking CUDA availability..."
    if command -v nvidia-smi &> /dev/null; then
        echo "✓ NVIDIA drivers detected"
        nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits
    else
        echo "✗ NVIDIA drivers not found. Please install NVIDIA drivers."
        exit 1
    fi
    
    if python -c "import torch; print(f'✓ PyTorch CUDA available: {torch.cuda.is_available()}')"; then
        python -c "import torch; print(f'✓ CUDA version: {torch.version.cuda}')"
        python -c "import torch; print(f'✓ GPU count: {torch.cuda.device_count()}')"
    else
        echo "✗ PyTorch CUDA not available. Please install PyTorch with CUDA support."
        exit 1
    fi
}

# Function to install base requirements
install_base_requirements() {
    echo "Installing base requirements..."
    
    # Backup original requirements if exists
    if [ -f "requirements.txt" ]; then
        cp requirements.txt requirements_original_backup.txt
        echo "✓ Backed up original requirements.txt"
    fi
    
    # Install from new requirements
    pip install -r requirements_axolotl.txt
    echo "✓ Base requirements installed"
}

# Function to create necessary directories
create_directories() {
    echo "Creating necessary directories..."
    
    mkdir -p data/images
    mkdir -p output/axolotl_minicpm_lora
    mkdir -p logs
    
    echo "✓ Directories created"
}

# Function to validate configuration files
validate_configs() {
    echo "Validating configuration files..."
    
    # Check YAML syntax
    if python -c "import yaml; yaml.safe_load(open('minicpm_axolotl_config.yaml'))"; then
        echo "✓ Axolotl config YAML is valid"
    else
        echo "✗ Axolotl config YAML has syntax errors"
        exit 1
    fi
    
    # Check JSON syntax
    if python -c "import json; json.load(open('configs/deepspeed_zero2_axolotl.json'))"; then
        echo "✓ DeepSpeed config JSON is valid"
    else
        echo "✗ DeepSpeed config JSON has syntax errors"
        exit 1
    fi
    
    echo "✓ Configuration files validated"
}

# Function to test dataset conversion (dry run)
test_dataset_conversion() {
    echo "Testing dataset conversion script..."
    
    if python -c "
import sys
sys.path.append('.')
from convert_dataset_axolotl import DatasetConverter
print('✓ Dataset conversion script can be imported')
"; then
        echo "✓ Dataset conversion script is ready"
    else
        echo "✗ Dataset conversion script has issues"
        exit 1
    fi
}

# Function to display next steps
display_next_steps() {
    echo ""
    echo "========================================="
    echo "Setup Complete! Next Steps:"
    echo "========================================="
    echo ""
    echo "1. Prepare your training data:"
    echo "   - Place your training data at: ./data/train_data.json"
    echo "   - Place your images at: ./data/images/"
    echo ""
    echo "2. Convert dataset to Axolotl format:"
    echo "   python convert_dataset_axolotl.py \\"
    echo "       --input ./data/train_data.json \\"
    echo "       --output ./data/train_data_axolotl.json \\"
    echo "       --image_base_path ./data/images"
    echo ""
    echo "3. (Optional) Test the setup:"
    echo "   ./train_axolotl.sh --dry-run"
    echo ""
    echo "4. Start training:"
    echo "   ./train_axolotl.sh"
    echo ""
    echo "5. Monitor training:"
    echo "   - TensorBoard: http://localhost:6006"
    echo "   - GPU usage: watch -n 1 nvidia-smi"
    echo ""
    echo "For detailed information, see MIGRATION.md"
    echo ""
}

# Main execution
main() {
    check_python
    check_cuda
    install_base_requirements
    create_directories
    validate_configs
    test_dataset_conversion
    display_next_steps
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-install)
            SKIP_INSTALL=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --skip-install    Skip pip install of requirements"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [ "$SKIP_INSTALL" = true ]; then
    echo "Skipping requirements installation..."
    check_python
    check_cuda
    create_directories
    validate_configs
    test_dataset_conversion
    display_next_steps
else
    main
fi