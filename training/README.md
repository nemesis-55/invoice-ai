# LLaMA-Factory Fine-tuning for MiniCPM-V 4.5 Invoice Extraction

This directory contains configuration and scripts for fine-tuning MiniCPM-V 4.5 on invoice extraction tasks using LLaMA-Factory with incremental LoRA training.

## Overview

MiniCPM-V 4.5 is a vision-language model built on Qwen3-8B LLM backbone with SigLIP2-400M vision encoder. This setup enables incremental LoRA training, allowing you to continue training from previous checkpoints without starting from scratch.

## Prerequisites

1. Install training dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Hugging Face authentication:
```bash
export HF_TOKEN=your_huggingface_token
huggingface-cli login
```

3. Ensure you have sufficient GPU memory (minimum 24GB VRAM recommended for batch_size=1)

## Data Preparation

### ShareGPT Format

LLaMA-Factory expects data in ShareGPT format. For invoice extraction, your training data should be in `data/invoice_train.json`:

```json
[
  {
    "conversations": [
      {
        "role": "user",
        "content": "Extract the following fields from the invoice image and return a JSON object:\n- OrderNumber\n- InvoiceNumber\n..."
      },
      {
        "role": "assistant",
        "content": "{\"OrderNumber\": \"12345\", \"InvoiceNumber\": \"INV-001\", ...}"
      }
    ],
    "image": "path/to/invoice_image.png"
  }
]
```

### Dataset Structure

```
training/
├── data/
│   └── invoice_train.json     # Your training data in ShareGPT format
├── output/                     # Training outputs (created automatically)
│   └── minicpmv45_invoice_lora/
│       ├── checkpoint-500/
│       ├── checkpoint-1000/
│       └── ...
```

### Data Format Requirements

- **conversations**: List of message objects with `role` and `content`
  - `role`: Either "user" or "assistant"
  - `content`: Text content (can include structured prompts)
- **image**: Path to image file (relative to data directory) or image URL

## Training

### Initial Training

To start training from the base MiniCPM-V 4.5 model:

```bash
cd training
./run_incremental_train.sh
```

Or use LLaMA-Factory CLI directly:

```bash
llamafactory-cli train llamafactory_config.yaml
```

### Incremental Training (Resume from Checkpoint)

To continue training from a previous LoRA checkpoint:

```bash
./run_incremental_train.sh output/minicpmv45_invoice_lora/checkpoint-1000
```

This will load the LoRA weights from the specified checkpoint and continue training, allowing you to incrementally improve your model.

### Configuration

Key training parameters in `llamafactory_config.yaml`:

- **resume_lora_training: true** - Enables incremental training from previous checkpoints
- **lora_rank: 64** - LoRA rank (higher = more parameters, better quality, slower training)
- **learning_rate: 2.0e-5** - Learning rate for fine-tuning
- **num_train_epochs: 3.0** - Number of epochs to train
- **per_device_train_batch_size: 1** - Batch size per GPU (increase if you have more VRAM)
- **gradient_accumulation_steps: 8** - Effective batch size = batch_size × accumulation_steps
- **save_steps: 500** - Save checkpoint every N steps
- **eval_steps: 500** - Evaluate every N steps

### Multi-GPU Training

For multi-GPU training, use:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train llamafactory_config.yaml
```

Or with DeepSpeed:

```bash
deepspeed --num_gpus=4 --master_port=9901 \
    $(which llamafactory-cli) train llamafactory_config.yaml \
    --deepspeed deepspeed_config.json
```

## Monitoring Training

LLaMA-Factory provides several ways to monitor training:

1. **Console Output**: Training loss, learning rate, and progress
2. **TensorBoard**: Visualize training metrics
   ```bash
   tensorboard --logdir=output/minicpmv45_invoice_lora
   ```
3. **WandB** (optional): Add `use_wandb: true` to config

## Evaluation

During training, validation is performed every `eval_steps` (default: 500) on `val_size` (default: 5%) of your data.

To evaluate a specific checkpoint:

```bash
llamafactory-cli eval llamafactory_config.yaml \
    --adapter_name_or_path output/minicpmv45_invoice_lora/checkpoint-1000
```

## Merging LoRA Adapters

After training, you can merge the LoRA adapter with the base model for faster inference:

### Using LLaMA-Factory

```bash
llamafactory-cli export llamafactory_config.yaml \
    --adapter_name_or_path output/minicpmv45_invoice_lora/checkpoint-final \
    --export_dir output/merged_model \
    --export_size 2 \
    --export_device cpu \
    --export_legacy_format false
```

### Manual Merging (Python)

```python
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModel.from_pretrained(
    "openbmb/MiniCPM-V-4_5",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "output/minicpmv45_invoice_lora/checkpoint-final"
)

# Merge and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained("output/merged_model")

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-V-4_5")
tokenizer.save_pretrained("output/merged_model")
```

## Deployment

### Using LoRA Adapter (Recommended for Development)

Update your handler.py to load the adapter:

```python
from peft import PeftModel

# Load base model
model = AutoModel.from_pretrained(
    "openbmb/MiniCPM-V-4_5",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    model,
    "output/minicpmv45_invoice_lora/checkpoint-final"
)
```

### Using Merged Model (Recommended for Production)

Update the `MODEL_ADAPTOR` environment variable:

```bash
export MODEL_ADAPTOR=output/merged_model
```

Or update directly in handler.py:

```python
adaptor_type_env = os.getenv("MODEL_ADAPTOR", "output/merged_model").strip()
```

## Hyperparameter Tuning

For better results, consider tuning:

1. **Learning Rate**: Try values between 1e-5 and 5e-5
2. **LoRA Rank**: Higher ranks (128, 256) may improve quality but increase training time
3. **Batch Size**: Increase if you have more VRAM (effective_batch_size = batch_size × accumulation_steps)
4. **Epochs**: More epochs may improve performance but watch for overfitting
5. **Warmup Ratio**: Adjust between 0.05 and 0.15

## Troubleshooting

### Out of Memory (OOM)

- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Use DeepSpeed ZeRO-3 for larger models
- Reduce `cutoff_len` (max sequence length)

### Poor Performance

- Increase `num_train_epochs` or `max_steps`
- Adjust `learning_rate` (try 1e-5 or 5e-5)
- Increase `lora_rank` for more model capacity
- Ensure training data is high quality and diverse
- Check for data imbalance

### Checkpoint Loading Issues

- Verify checkpoint path is correct
- Ensure LoRA configuration matches (rank, alpha, target modules)
- Check that base model matches checkpoint's base model

## Advanced Features

### Deep Thinking Mode Training

To train the model with deep thinking capability, include system messages in your training data:

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "You are a helpful assistant. Think step by step carefully before responding."
    },
    {
      "role": "user",
      "content": "Extract invoice information..."
    },
    {
      "role": "assistant",
      "content": "Let me analyze this step by step...\n\nStep 1: Identify the invoice number...\n\n{\"OrderNumber\": ...}"
    }
  ],
  "image": "invoice.png"
}
```

### Custom LoRA Targets

By default, `lora_target: all` applies LoRA to all linear layers. For more control:

```yaml
lora_target: q_proj,v_proj,k_proj,o_proj  # Attention layers only
```

### Quantization (QLoRA)

For training on smaller GPUs, enable 4-bit quantization:

```yaml
quantization_bit: 4
quantization_type: nf4
double_quantization: true
```

## References

- [MiniCPM-V 4.5 Model Card](https://huggingface.co/openbmb/MiniCPM-V-4_5)
- [LLaMA-Factory Documentation](https://github.com/hiyouga/LLaMA-Factory)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [ShareGPT Data Format](https://github.com/hiyouga/LLaMA-Factory#data-format)

## License

This training setup follows the license of MiniCPM-V 4.5 and LLaMA-Factory. Please review their respective licenses before commercial use.
