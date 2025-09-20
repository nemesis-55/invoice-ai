# invoice-ai

## Configuration

This repository supports configurable model precision and GPU usage for inference.

### Default Configuration (Recommended)
- **Precision**: 16-bit (bfloat16) for high-quality inference
- **GPU Usage**: Single GPU to avoid multi-GPU complexity

### Environment Variables
- `MODEL_PRECISION`: `16bit` (default), `8bit`, or `4bit`
- `GPU_DEVICE`: `single` (default), `auto`, or specific device like `cuda:1`

See [CONFIG.md](CONFIG.md) for detailed configuration options.

### Quick Start
```bash
# Use defaults (16-bit, single GPU)
python handler.py

# Use 8-bit quantization  
MODEL_PRECISION=8bit python handler.py

# Use auto GPU distribution
GPU_DEVICE=auto python handler.py
```
