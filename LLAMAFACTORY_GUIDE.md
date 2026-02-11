# LLamaFactory Training Guide for MiniCPM-V-4.5

This guide explains how to use LLamaFactory for training the MiniCPM-V-4.5 model on invoice extraction tasks.

## Table of Contents
1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Training Options](#training-options)
4. [Configuration Files](#configuration-files)
5. [Training Commands](#training-commands)
6. [Web UI](#web-ui)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ or 12.1+
- GPU with at least 24GB VRAM (for LoRA) or 80GB+ (for full fine-tuning)

### Install LLamaFactory

```bash
# Clone LLamaFactory repository
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# Install with MiniCPM-V support
pip install -e ".[torch,metrics,deepspeed,minicpm_v]"

# Or install from PyPI
pip install llamafactory[torch,metrics,deepspeed,minicpm_v]
```

### Verify Installation

```bash
llamafactory-cli version
```

---

## Dataset Preparation

### Current Format (Compatible with LLamaFactory)

Our existing dataset format is already compatible with LLamaFactory:

```json
[
  {
    "id": "185486_1",
    "images": ["path/to/image.jpg"],
    "conversations": [
      {"from": "human", "value": "<image>\nExtract fields from the invoice..."},
      {"from": "gpt", "value": "{...extracted JSON...}"}
    ]
  }
]
```

### Dataset Registration

The dataset is registered in `llamafactory_configs/dataset_info.json`:

```json
{
  "invoice_training": {
    "file_name": "data/train_data.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "images": "images"
    }
  }
}
```

**Note:** The training scripts automatically copy this file to LLamaFactory's data directory.

---

## Training Options

### Option 1: LoRA Fine-tuning (Recommended)

**Advantages:**
- Memory efficient (~24GB VRAM on single GPU)
- Faster training
- Easier to merge adapters
- Good for most use cases

**Use Case:** When you want to adapt the model to invoice extraction with limited resources.

**Command:**
```bash
cd /path/to/invoice-ai
bash training/train_llamafactory_lora.sh
```

### Option 2: Full Fine-tuning

**Advantages:**
- Maximum model adaptation
- Better performance on complex tasks
- Full control over all parameters

**Requirements:**
- Multiple GPUs (2+ recommended)
- 80GB+ VRAM total
- DeepSpeed ZeRO-2 or ZeRO-3

**Use Case:** When you have sufficient compute resources and need maximum performance.

**Command:**
```bash
cd /path/to/invoice-ai
bash training/train_llamafactory_full.sh
```

---

## Configuration Files

### LoRA Configuration (`llamafactory_configs/minicpm_v45_lora.yaml`)

Key parameters:
- **model_name_or_path:** `openbmb/MiniCPM-V-4_5`
- **finetuning_type:** `lora`
- **lora_rank:** 64 (balance between quality and efficiency)
- **learning_rate:** 5.0e-6 (higher for LoRA)
- **batch_size:** 1 with gradient_accumulation_steps=4
- **epochs:** 3

### Full Fine-tuning Configuration (`llamafactory_configs/minicpm_v45_full.yaml`)

Key parameters:
- **model_name_or_path:** `openbmb/MiniCPM-V-4_5`
- **finetuning_type:** `full`
- **learning_rate:** 1.0e-6 (lower for full fine-tuning)
- **batch_size:** 1 with gradient_accumulation_steps=8
- **deepspeed:** `training/ds_config_zero2.json`
- **epochs:** 3

---

## Training Commands

### Using Shell Scripts (Easiest)

```bash
# LoRA training
bash training/train_llamafactory_lora.sh

# Full fine-tuning
bash training/train_llamafactory_full.sh
```

### Using LLamaFactory CLI Directly

```bash
# LoRA training
llamafactory-cli train llamafactory_configs/minicpm_v45_lora.yaml

# Full fine-tuning
llamafactory-cli train llamafactory_configs/minicpm_v45_full.yaml
```

### Using Python API

```python
from llamafactory.train import run_train

run_train(
    config_path="llamafactory_configs/minicpm_v45_lora.yaml"
)
```

### Multi-GPU Training

LLamaFactory automatically detects and uses all available GPUs. For explicit control:

```bash
# Specify number of GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train llamafactory_configs/minicpm_v45_lora.yaml

# Or use torchrun for more control
torchrun --nproc_per_node=4 \
    -m llamafactory.train \
    llamafactory_configs/minicpm_v45_lora.yaml
```

---

## Web UI

LLamaFactory includes a web-based UI for easier configuration and training:

### Launch Web UI

```bash
llamafactory-cli webui
```

This will start a local web server (usually at `http://localhost:7860`).

### Using the Web UI

1. **Select Model:** Choose `openbmb/MiniCPM-V-4_5`
2. **Select Dataset:** Choose `invoice_training`
3. **Configure Training:**
   - Set training method (LoRA/Full)
   - Adjust hyperparameters
   - Set output directory
4. **Start Training:** Click "Start Training"
5. **Monitor Progress:** View real-time training metrics

### Preview Dataset

The Web UI also allows you to preview your dataset before training.

---

## Monitoring Training

### TensorBoard

Training logs are automatically saved for TensorBoard:

```bash
# LoRA training logs
tensorboard --logdir output/minicpm_v45_lora_invoice/logs

# Full fine-tuning logs
tensorboard --logdir output/minicpm_v45_full_invoice/logs
```

### LlamaBoard (LLamaFactory's Built-in Monitor)

```bash
llamafactory-cli board
```

---

## Inference with Fine-tuned Model

### Load LoRA Adapter

```python
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

# Load base model
model = AutoModel.from_pretrained(
    "openbmb/MiniCPM-V-4_5",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    model,
    "output/minicpm_v45_lora_invoice"
)

tokenizer = AutoTokenizer.from_pretrained(
    "openbmb/MiniCPM-V-4_5",
    trust_remote_code=True
)

# Inference
image = Image.open("invoice.jpg")
prompt = "<image>\nExtract the following fields..."
response = model.chat(image, prompt, tokenizer)
```

### Merge LoRA Weights (Optional)

For deployment, you can merge LoRA weights into the base model:

```bash
llamafactory-cli export \
    --model_name_or_path openbmb/MiniCPM-V-4_5 \
    --adapter_name_or_path output/minicpm_v45_lora_invoice \
    --template minicpm_v \
    --export_dir output/minicpm_v45_merged \
    --export_size 2 \
    --export_legacy_format false
```

---

## Troubleshooting

### Out of Memory (OOM) Errors

**Solutions:**
1. Reduce batch size: `per_device_train_batch_size: 1`
2. Increase gradient accumulation: `gradient_accumulation_steps: 8`
3. Enable gradient checkpointing: `gradient_checkpointing: true`
4. Use DeepSpeed ZeRO-3: `deepspeed: training/ds_config_zero3.json`
5. Use smaller LoRA rank: `lora_rank: 32`

### Dataset Not Found

**Error:** `ValueError: Dataset 'invoice_training' not found`

**Solution:** Ensure `dataset_info.json` is copied to LLamaFactory's data directory:
```bash
cp llamafactory_configs/dataset_info.json \
   $(python -c "import llamafactory, os; print(os.path.join(os.path.dirname(llamafactory.__file__), 'data'))")/dataset_info.json
```

### Training Crashes Early

**Possible Causes:**
1. Corrupted images in dataset
2. Incompatible image formats
3. Missing image files

**Solution:**
```python
# Validate all images exist and are readable
import json
from PIL import Image

with open("data/train_data.json") as f:
    data = json.load(f)

for item in data:
    try:
        img = Image.open(item["image"])
        img.verify()
    except Exception as e:
        print(f"Bad image: {item['image']} - {e}")
```

### Slow Training

**Solutions:**
1. Enable mixed precision: `bf16: true` or `fp16: true`
2. Increase number of workers: `preprocessing_num_workers: 16`
3. Use faster data loading: Set `dataloader_num_workers: 4`
4. Optimize dataset: Pre-process and cache images

### Model Loading Errors

**Error:** `Trust remote code error`

**Solution:** Ensure you're using the latest transformers:
```bash
pip install --upgrade transformers>=4.44.0
```

---

## Comparison: Native Scripts vs LLamaFactory

| Feature | Native Scripts | LLamaFactory |
|---------|---------------|--------------|
| Setup Complexity | Medium | Easy |
| Configuration | Shell scripts | YAML files |
| Web UI | No | Yes ✓ |
| Monitoring | TensorBoard only | TensorBoard + LlamaBoard |
| Dataset Preview | No | Yes ✓ |
| Model Export | Manual | Built-in CLI |
| Community Support | Limited | Active community |
| Flexibility | High (custom code) | Medium (config-based) |

---

## Best Practices

1. **Start with LoRA:** Always try LoRA fine-tuning first before full fine-tuning
2. **Validate Dataset:** Check all images are valid before training
3. **Monitor Closely:** Watch training loss for overfitting
4. **Save Checkpoints:** Keep multiple checkpoints to find the best one
5. **Use Validation Set:** Always use a validation set to monitor generalization
6. **Gradual Learning:** Start with lower learning rate and adjust if needed

---

## Resources

- [LLamaFactory GitHub](https://github.com/hiyouga/LLaMA-Factory)
- [MiniCPM-V Documentation](https://minicpm-o.readthedocs.io/)
- [MiniCPM-V-4.5 Model Card](https://huggingface.co/openbmb/MiniCPM-V-4_5)
- [LLamaFactory Fine-tuning Guide](https://minicpm-o.readthedocs.io/en/latest/finetune/llamafactory.html)

---

## Support

For issues or questions:
1. Check this guide first
2. Review LLamaFactory documentation
3. Check GitHub issues for similar problems
4. Open a new issue with:
   - Error message
   - Configuration file
   - Training command used
   - System specs (GPU, VRAM, CUDA version)
