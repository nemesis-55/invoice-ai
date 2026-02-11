# invoice-ai

AI-powered invoice extraction system using **MiniCPM-V-4.5** vision-language model for extracting structured data from invoice images.

## 🚀 Recent Updates

**[2026-02-11] Major Model Upgrade:**
- ✨ Upgraded from MiniCPM-V-2.6 to **MiniCPM-V-4.5**
- 🎯 Improved OCR and document parsing capabilities
- 🔧 Added **LLamaFactory** integration for streamlined training
- 📚 Comprehensive documentation and training guides

## Features

- **Advanced Vision-Language Model**: Uses MiniCPM-V-4.5 with Qwen3-8B backbone
- **High Accuracy OCR**: Superior document understanding and field extraction
- **Flexible Training**: Support for both native scripts and LLamaFactory
- **Efficient Fine-tuning**: LoRA and full fine-tuning options
- **DeepSpeed Integration**: Distributed training with ZeRO optimization
- **Production Ready**: Containerized deployment with RunPod support

## Model Specifications

- **Base Model**: openbmb/MiniCPM-V-4_5
- **Architecture**: Qwen3-8B LLM + SigLIP2-400M vision encoder
- **Parameters**: 8B total
- **Key Capabilities**:
  - High-resolution image processing (up to 1.8M pixels)
  - Advanced token compression (up to 96x for video)
  - 30+ languages support
  - State-of-the-art OCR performance

## Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/Arindam2002/invoice-ai.git
cd invoice-ai

# Install dependencies
pip install -r requirements.txt

# (Optional) Install LLamaFactory for enhanced training
pip install llamafactory[torch,metrics,deepspeed,minicpm_v]
```

### RunPod Setup (Cloud GPU Training)

```bash
# On your RunPod pod terminal
cd /workspace
bash -c "$(curl -fsSL https://raw.githubusercontent.com/Arindam2002/invoice-ai/main/runpod_setup.sh)"

# Or manually:
git clone https://github.com/Arindam2002/invoice-ai.git
cd invoice-ai
bash runpod_setup.sh
```

**See [RUNPOD_TRAINING_GUIDE.md](RUNPOD_TRAINING_GUIDE.md) for complete RunPod training instructions and GPU recommendations.**

### Training Options

#### Option 1: Native Training Scripts

```bash
# LoRA fine-tuning
bash training/finetune_lora.sh

# Full fine-tuning with DeepSpeed
bash training/finetune_ds.sh
```

#### Option 2: LLamaFactory Training (Recommended)

```bash
# LoRA training with LLamaFactory
bash training/train_llamafactory_lora.sh

# Full fine-tuning with LLamaFactory
bash training/train_llamafactory_full.sh

# Or use the Web UI
llamafactory-cli webui
```

#### Option 3: RunPod Cloud Training

```bash
# On RunPod pod with network volume mounted
# LoRA training (optimized for RunPod)
bash training/runpod_train_lora.sh

# Full fine-tuning (multi-GPU)
bash training/runpod_train_full.sh
```

**See [RUNPOD_TRAINING_GUIDE.md](RUNPOD_TRAINING_GUIDE.md) for:**
- GPU configuration recommendations
- Cost optimization strategies
- Step-by-step setup instructions

## Documentation

- **[README.md](README.md)** - This file: Quick start and overview
- **[RUNPOD_TRAINING_GUIDE.md](RUNPOD_TRAINING_GUIDE.md)** - ⭐ Complete guide for training on RunPod (GPU configs, setup, costs)
- **[UPGRADE_NOTES.md](UPGRADE_NOTES.md)** - Detailed migration guide from V-2.6 to V-4.5
- **[LLAMAFACTORY_GUIDE.md](LLAMAFACTORY_GUIDE.md)** - Complete LLamaFactory training guide
- **[MIGRATION_CHECKLIST.md](MIGRATION_CHECKLIST.md)** - Deployment checklist
- **Training Scripts**: Located in `training/` directory
- **Configuration Files**: Located in `llamafactory_configs/` directory

## Dataset Preparation

```bash
# Set environment variables
export RAW_DATA_OUTPUT="path/to/raw_data.json"
export TRAIN_DATA_PATH="data/train_data.json"

# Run data preparation
python prepare_data/create_training_data.py
```

**Note:** The script creates a single `train_data.json` file. LLamaFactory automatically handles train/validation splitting during training based on the `val_size` parameter (default: 0.1 = 10% validation).

## Project Structure

```
invoice-ai/
├── training/                      # Training scripts and configurations
│   ├── finetune.py               # Main training script
│   ├── finetune_lora.sh          # LoRA training (native)
│   ├── finetune_ds.sh            # DeepSpeed training (native)
│   ├── train_llamafactory_lora.sh # LoRA training (LLamaFactory)
│   ├── train_llamafactory_full.sh # Full training (LLamaFactory)
│   ├── dataset.py                # Dataset handling
│   ├── trainer.py                # Custom trainer
│   └── ds_config_zero*.json      # DeepSpeed configurations
├── llamafactory_configs/          # LLamaFactory configurations
│   ├── dataset_info.json         # Dataset registration
│   ├── minicpm_v45_lora.yaml     # LoRA config
│   └── minicpm_v45_full.yaml     # Full fine-tuning config
├── prepare_data/                  # Dataset preparation scripts
│   ├── create_training_data.py   # Convert raw to training format
│   └── create_raw_data.py        # Initial data processing
├── models/                        # Model schemas
│   └── payloads/                 # API payload definitions
├── helper/                        # Utility functions
├── handler.py                     # Inference handler
├── UPGRADE_NOTES.md              # Migration documentation
├── LLAMAFACTORY_GUIDE.md         # LLamaFactory guide
└── requirements.txt              # Python dependencies
```

## Training Comparison

| Feature | Native Scripts | LLamaFactory |
|---------|---------------|--------------|
| Setup | Medium complexity | Easy |
| Configuration | Shell scripts | YAML files |
| Web UI | ❌ | ✅ |
| Monitoring | TensorBoard | TensorBoard + LlamaBoard |
| Dataset Preview | ❌ | ✅ |
| Model Export | Manual | Built-in CLI |
| Recommended For | Advanced users | All users |

## Hardware Requirements

### Local Training

#### LoRA Fine-tuning
- **Minimum**: 1x GPU with 24GB VRAM (RTX 4090, RTX A6000)
- **Recommended**: 1x GPU with 40GB VRAM (A100, A40)
- **Training Time**: ~8-12 hours for 10K steps

#### Full Fine-tuning
- **Minimum**: 2x GPU with 40GB VRAM (with DeepSpeed ZeRO-2)
- **Recommended**: 4x GPU with 80GB VRAM (A100)
- **Training Time**: ~24-48 hours for 10K steps

### RunPod Cloud Training (Recommended)

See **[RUNPOD_TRAINING_GUIDE.md](RUNPOD_TRAINING_GUIDE.md)** for detailed configurations:

| Training Type | GPUs | VRAM | Cost/hr (Spot) | Use Case |
|--------------|------|------|----------------|----------|
| **Budget LoRA** | 1x RTX 4090 | 24GB | $0.30-0.50 | Experiments |
| **Recommended LoRA** | 2x A40 | 96GB | $0.60-1.00 | Production |
| **Full Fine-tuning** | 2x A100 80GB | 160GB | $2.00-4.00 | Best quality |
| **Large-scale** | 4x A100 80GB | 320GB | $4.00-8.00 | Research |

**Cost Examples:**
- Quick LoRA experiment: ~$1-2 for 2 hours
- Production LoRA training: ~$6-10 for 8 hours
- Full fine-tuning: ~$30-40 for 12-15 hours

## Inference

```python
from transformers import AutoModel, AutoTokenizer
from PIL import Image

# Load model and tokenizer
model = AutoModel.from_pretrained(
    "path/to/fine-tuned-model",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "path/to/fine-tuned-model",
    trust_remote_code=True
)

# Load invoice image
image = Image.open("invoice.jpg")

# Extract information
prompt = "<image>\nExtract the following fields from the invoice..."
response = model.chat(image, prompt, tokenizer)
print(response)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

## Acknowledgments

- [OpenBMB](https://github.com/OpenBMB) for MiniCPM-V models
- [LLamaFactory](https://github.com/hiyouga/LLaMA-Factory) for training framework
- [HuggingFace](https://huggingface.co/) for model hosting and transformers library

## Support

For issues, questions, or feature requests, please open an issue on GitHub.

---

**Made with ❤️ by the Invoice-AI Team**
