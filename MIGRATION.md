# Migration Guide: Custom MiniCPM Training → Axolotl Framework

This document explains how to migrate from the existing custom PyTorch training setup to the Axolotl framework for improved performance and memory efficiency on dual RTX 5090 GPUs.

## Table of Contents
- [Overview](#overview)
- [Key Differences](#key-differences)
- [Performance Benefits](#performance-benefits)
- [Migration Steps](#migration-steps)
- [Configuration Comparison](#configuration-comparison)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Overview

### Current Setup (Before Migration)
- **Training Script**: `training/finetune.py` with custom `CPMTrainer`
- **Launch Script**: `training/finetune_lora.sh` using `torchrun`
- **Dataset**: Custom `SupervisedDataset` in `training/dataset.py`
- **Configuration**: Command-line arguments + DeepSpeed JSON configs
- **Model**: MiniCPM-V-2_6 with manual LoRA setup

### New Setup (After Migration)
- **Framework**: Axolotl with YAML-based configuration
- **Training Script**: `train_axolotl.sh` with automated setup
- **Dataset**: Converted to Axolotl chat_template format
- **Configuration**: Single YAML file (`minicpm_axolotl_config.yaml`)
- **Model**: Same MiniCPM-V-2_6 with QLoRA optimization

## Key Differences

| Aspect | Current Setup | Axolotl Setup |
|--------|---------------|---------------|
| **Configuration** | Multiple files + CLI args | Single YAML file |
| **Dataset Format** | Custom conversations format | Chat template with base64 images |
| **Quantization** | 16-bit (optional) | 4-bit QLoRA by default |
| **Memory Usage** | ~24-28GB per GPU | ~18-22GB per GPU |
| **Setup Complexity** | Manual deepspeed + torchrun | Automated with accelerate |
| **Monitoring** | Basic tensorboard | Enhanced wandb + tensorboard |
| **Flash Attention** | Manual configuration | Built-in optimization |

## Performance Benefits

### Memory Efficiency
- **QLoRA**: Reduces memory usage by ~30-40% through 4-bit quantization
- **Optimized Attention**: Flash Attention 2 reduces memory overhead
- **Better Gradient Checkpointing**: More efficient implementation

### Training Speed
- **Optimized Kernels**: Better CUDA kernel utilization
- **Improved Communication**: Enhanced multi-GPU communication patterns
- **Faster Data Loading**: Optimized data pipeline

### Ease of Use
- **Unified Configuration**: Single YAML file instead of multiple scripts
- **Automated Setup**: Dependency management and environment setup
- **Better Monitoring**: Integrated wandb and tensorboard logging

## Migration Steps

### Step 1: Backup Current Setup
```bash
# Create backup of current training setup
cp -r training training_backup
cp requirements.txt requirements_original.txt
```

### Step 2: Install Axolotl Dependencies
```bash
# Run the automated installation script
./train_axolotl.sh --dry-run  # Validate setup first
./train_axolotl.sh --skip-install  # If dependencies already installed
```

### Step 3: Convert Dataset
```bash
# Convert your existing dataset to Axolotl format
python convert_dataset_axolotl.py \
    --input ./data/train_data.json \
    --output ./data/train_data_axolotl.json \
    --image_base_path ./data/images \
    --split_ratio 0.9
```

### Step 4: Update Configuration
The Axolotl configuration is in `minicpm_axolotl_config.yaml`. Key settings:

```yaml
base_model: openbmb/MiniCPM-V-2_6
adapter: qlora
lora_r: 64
lora_alpha: 128
sequence_len: 8192
micro_batch_size: 1
gradient_accumulation_steps: 4
```

### Step 5: Run Training
```bash
# Start training with dual GPU optimization
./train_axolotl.sh
```

## Configuration Comparison

### LoRA Configuration
**Current Setup** (`finetune_lora.sh`):
```bash
--lora_target_modules "llm\..*layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
--use_lora true
--tune_vision true
--tune_llm false
```

**Axolotl Setup** (`minicpm_axolotl_config.yaml`):
```yaml
adapter: qlora
lora_r: 64
lora_alpha: 128
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

### Training Parameters
**Current Setup**:
```bash
--per_device_train_batch_size 1
--gradient_accumulation_steps 1
--learning_rate 5e-6
--max_steps 10000
--bf16 true
```

**Axolotl Setup**:
```yaml
micro_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 5.0e-6
num_epochs: 3
bf16: auto
```

### DeepSpeed Configuration
Both setups use similar DeepSpeed Zero2 configuration, but Axolotl's version includes optimizations for the framework.

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
**Symptoms**: OOM errors during training
**Solutions**:
- Reduce `micro_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Enable `load_in_4bit: true`
- Use `bf16: true` instead of fp16

#### 2. Slow Training Speed
**Symptoms**: Training slower than expected
**Solutions**:
- Ensure `flash_attention: true` is enabled
- Check `tf32: true` for RTX 5090
- Verify dual GPU utilization with `nvidia-smi`
- Increase `gradient_accumulation_steps` for better GPU utilization

#### 3. Dataset Loading Issues
**Symptoms**: Errors loading converted dataset
**Solutions**:
- Verify image paths in original dataset
- Check base64 encoding in converted dataset
- Ensure sufficient disk space for base64 images
- Use `--image_base_path` correctly in conversion script

#### 4. Model Loading Errors
**Symptoms**: Model fails to load or initialize
**Solutions**:
- Ensure model name is correct: `openbmb/MiniCPM-V-2_6`
- Check internet connection for model download
- Verify `trust_remote_code: true` is set
- Clear HuggingFace cache if needed: `rm -rf ~/.cache/huggingface`

### Performance Tuning

#### For RTX 5090 Optimization
```yaml
# In minicpm_axolotl_config.yaml
tf32: true  # Enable TensorFloat-32 for RTX 5090
flash_attention: true
gradient_checkpointing: true
bf16: auto
load_in_4bit: true
```

#### For Maximum Memory Efficiency
```yaml
micro_batch_size: 1
gradient_accumulation_steps: 8  # Increase for effective batch size
load_in_4bit: true
bnb_4bit_use_double_quant: true
```

#### For Maximum Speed
```yaml
micro_batch_size: 2  # If memory allows
gradient_accumulation_steps: 2
flash_attention: true
tf32: true
dataloader_num_workers: 4
```

## Best Practices

### 1. Monitoring Training
- Use wandb for comprehensive tracking: `wandb_project: invoice-ai-axolotl`
- Monitor GPU utilization: `watch -n 1 nvidia-smi`
- Check tensorboard: `tensorboard --logdir=./output/axolotl_minicpm_lora`

### 2. Checkpointing Strategy
- Set `save_steps: 1000` for regular checkpoints
- Use `save_total_limit: 10` to manage disk space
- Enable `load_best_model_at_end: true` for best model selection

### 3. Data Management
- Keep original dataset for reference
- Validate converted dataset with sample output
- Use version control for dataset versions
- Monitor data loading performance

### 4. Experiment Tracking
- Use descriptive `wandb_run_id` names
- Tag experiments with configuration changes
- Track key metrics: loss, learning rate, GPU utilization
- Save configuration files with each run

### 5. Resource Management
- Monitor disk space (base64 images are larger)
- Use `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` for memory fragmentation
- Set appropriate `OMP_NUM_THREADS=8` for CPU utilization

## Validation

### Verify Migration Success
1. **Training Starts**: No immediate errors on launch
2. **GPU Utilization**: Both GPUs showing 80%+ utilization
3. **Memory Usage**: Within expected ranges (18-22GB per GPU)
4. **Loss Decrease**: Training loss decreases over time
5. **Checkpoints**: Model checkpoints save successfully

### Performance Benchmarks
Compare training metrics:
- **Steps per second**: Should be similar or better than original
- **Memory usage**: Should be reduced with QLoRA
- **Model quality**: Validate with evaluation metrics

## Support and Resources

### Documentation
- [Axolotl Documentation](https://github.com/OpenAccess-AI-Collective/axolotl)
- [MiniCPM Model Documentation](https://huggingface.co/openbmb/MiniCPM-V-2_6)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)

### Community
- [Axolotl Discord](https://discord.gg/HhrNrHJPRb)
- [Axolotl GitHub Issues](https://github.com/OpenAccess-AI-Collective/axolotl/issues)

### Rollback Plan
If migration encounters issues:
1. Use backup training setup: `cp -r training_backup training`
2. Restore original requirements: `cp requirements_original.txt requirements.txt`
3. Continue with original training pipeline
4. Report issues for future migration attempts