# Configuration Guide

## Environment Variables

### Model Precision Configuration
- **MODEL_PRECISION**: Controls the precision/quantization level for model inference
  - `16bit` (default): Uses bfloat16 precision - best quality, higher memory usage
  - `8bit`: Uses 8-bit quantization - balanced quality and memory 
  - `4bit`: Uses 4-bit quantization - lowest memory usage, may reduce quality
  
### GPU Configuration  
- **GPU_DEVICE**: Controls GPU usage for model inference
  - `single` (default): Uses only the first GPU (cuda:0) - recommended for single GPU setups
  - `auto`: Automatically distributes model across all available GPUs
  - `cuda:0`, `cuda:1`, etc.: Specify exact GPU device to use

## Default Configuration (16-bit, Single GPU)

The default settings are optimized for the user's requirements:
- **Precision**: 16-bit (bfloat16) for high quality inference
- **GPU Usage**: Single GPU (cuda:0) to avoid multi-GPU complexity
- **Memory Efficient**: Reasonable memory usage for single GPU inference

## Example Usage

### Using defaults (16-bit, single GPU):
```bash
python handler.py
```

### Using 8-bit quantization on auto GPU distribution:
```bash
MODEL_PRECISION=8bit GPU_DEVICE=auto python handler.py
```

### Using specific GPU:
```bash
GPU_DEVICE=cuda:1 python handler.py
```

## Docker Configuration

Set environment variables in your Docker run command:
```bash
docker run -e MODEL_PRECISION=16bit -e GPU_DEVICE=single your-image
```

Or in docker-compose.yml:
```yaml
environment:
  - MODEL_PRECISION=16bit
  - GPU_DEVICE=single
```

## Memory Usage Guidelines

| Precision | Approximate VRAM Usage | Quality | Use Case |
|-----------|----------------------|---------|----------|
| 16bit     | ~12-16GB             | Highest | Production inference, quality-critical |
| 8bit      | ~8-12GB              | High    | Balanced performance/memory |  
| 4bit      | ~6-8GB               | Good    | Memory-constrained environments |

## Training vs Inference

Note: This configuration only affects **inference** (handler.py). 
Training configuration is controlled separately via the Axolotl config (minicpm_axolotl_config.yaml) which uses 4-bit QLoRA for memory efficiency during training.