# Axolotl Training Setup for Invoice AI

This directory contains the migrated training setup using the Axolotl framework for improved performance and memory efficiency on dual RTX 5090 GPUs.

## Quick Start

### 1. Setup Environment
```bash
# Run the setup script to prepare environment
./setup_axolotl.sh

# Or if you want to skip automatic installation
./setup_axolotl.sh --skip-install
```

### 2. Prepare Data
```bash
# Convert your existing dataset to Axolotl format
python convert_dataset_axolotl.py \
    --input ./data/train_data.json \
    --output ./data/train_data_axolotl.json \
    --image_base_path ./data/images \
    --split_ratio 0.9
```

### 3. Start Training
```bash
# Test setup (recommended first)
./train_axolotl.sh --dry-run

# Start actual training
./train_axolotl.sh
```

## Key Files

- **`minicpm_axolotl_config.yaml`** - Main Axolotl configuration
- **`train_axolotl.sh`** - Training script with dual GPU optimization
- **`convert_dataset_axolotl.py`** - Dataset conversion utility
- **`requirements_axolotl.txt`** - Updated dependencies
- **`MIGRATION.md`** - Detailed migration guide

## Configuration

### Model Settings
- **Base Model**: `openbmb/MiniCPM-V-2_6`
- **Adapter**: QLoRA for memory efficiency
- **LoRA Rank**: 64 (preserving original settings)
- **Quantization**: 4-bit for RTX 5090 optimization

### Training Settings
- **Batch Size**: 1 per device
- **Gradient Accumulation**: 4 steps
- **Learning Rate**: 5e-6
- **Max Length**: 8192 tokens
- **Mixed Precision**: bfloat16

### Hardware Optimization
- **Flash Attention 2**: Enabled for speed
- **Gradient Checkpointing**: Enabled for memory
- **TensorFloat-32**: Enabled for RTX 5090
- **DeepSpeed ZeRO-2**: For distributed training

## Monitoring

### TensorBoard
```bash
tensorboard --logdir=./output/axolotl_minicpm_lora --port=6006
```

### GPU Monitoring
```bash
watch -n 1 nvidia-smi
```

### Weights & Biases (Optional)
Configure in `minicpm_axolotl_config.yaml`:
```yaml
wandb_project: invoice-ai-axolotl
wandb_watch: gradients
```

## Performance Benefits

Compared to the original setup:
- **Memory**: ~30-40% reduction (QLoRA + optimizations)
- **Speed**: ~20-30% improvement (Flash Attention + optimized kernels)
- **Ease of Use**: Single configuration file vs multiple scripts
- **Monitoring**: Enhanced logging and tracking

## Troubleshooting

### Common Issues

1. **CUDA OOM**: Reduce `micro_batch_size` or increase `gradient_accumulation_steps`
2. **Slow Training**: Ensure `flash_attention: true` and `tf32: true`
3. **Dataset Errors**: Check image paths and base64 encoding
4. **Model Loading**: Verify internet connection and HuggingFace cache

See `MIGRATION.md` for detailed troubleshooting guide.

## Directory Structure

```
.
├── minicpm_axolotl_config.yaml          # Main config
├── train_axolotl.sh                     # Training script
├── convert_dataset_axolotl.py           # Dataset converter
├── setup_axolotl.sh                     # Setup script
├── requirements_axolotl.txt             # Dependencies
├── MIGRATION.md                         # Migration guide
├── configs/
│   ├── deepspeed_zero2_axolotl.json    # DeepSpeed config
│   └── accelerate_config.yaml          # Accelerate config
├── data/
│   ├── train_data.json                 # Original dataset
│   ├── train_data_axolotl.json         # Converted dataset
│   └── images/                         # Training images
└── output/
    └── axolotl_minicpm_lora/           # Training outputs
```

## Support

- **Documentation**: See `MIGRATION.md` for detailed information
- **Issues**: Check the troubleshooting section first
- **Performance**: Monitor GPU utilization and memory usage
- **Validation**: Use `--dry-run` to test configuration before training