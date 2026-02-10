# LLaMA-Factory Incremental Training for MiniCPM-V 4.5

This directory contains configuration and scripts for fine-tuning MiniCPM-V 4.5 using LLaMA-Factory with incremental LoRA training.

## Overview

LLaMA-Factory provides a streamlined framework for fine-tuning large language models with LoRA adapters. This setup enables:
- **Incremental training**: Resume training from previous checkpoints to build upon existing knowledge
- **Memory efficiency**: LoRA adapters require significantly less VRAM than full fine-tuning
- **Easy experimentation**: Quickly iterate on different hyperparameters and datasets

## Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your training data in ShareGPT format (see Data Preparation below)

## Data Preparation

Create a training dataset file `data/invoice_train.json` in ShareGPT format:

```json
[
  {
    "conversations": [
      {"role": "user", "content": "Extract invoice data from this image..."},
      {"role": "assistant", "content": "{\"OrderNumber\": \"12345\", ...}"}
    ],
    "image": "path/to/invoice_image.jpg"
  },
  ...
]
```

**Key points:**
- Images can be provided as file paths (relative to data directory) or base64-encoded strings
- The `conversations` field should contain alternating user/assistant messages
- For invoice extraction, structure the assistant's response as JSON matching your expected schema
- Include diverse examples: different invoice layouts, languages, edge cases

## Configuration

The main configuration is in `llamafactory_config.yaml`. Key parameters:

- **model_name_or_path**: Base model (`openbmb/MiniCPM-V-4_5`)
- **lora_rank**: LoRA rank (64) - higher = more capacity but slower training
- **learning_rate**: 2.0e-5 (adjust based on your dataset size)
- **num_train_epochs**: Number of complete passes through the dataset
- **per_device_train_batch_size**: 1 (increase if you have more VRAM)
- **gradient_accumulation_steps**: 8 (effective batch size = batch_size × accumulation_steps)

## Training

### Initial Training

Start training from the base model:

```bash
cd training
bash run_incremental_train.sh
```

This will:
1. Load the base MiniCPM-V 4.5 model
2. Apply LoRA adapters to trainable layers
3. Train on your dataset
4. Save checkpoints to `output/minicpmv45_invoice_lora/`

### Incremental Training

Resume training from a previous checkpoint to continue improving the model:

```bash
bash run_incremental_train.sh output/minicpmv45_invoice_lora/checkpoint-1000
```

This will:
1. Load the base model
2. Load the existing LoRA adapter from the checkpoint
3. Continue training, updating the adapter weights
4. Save new checkpoints

**Use cases for incremental training:**
- Add new invoice types or layouts to your model
- Improve performance on edge cases
- Adapt the model to new requirements without starting from scratch

## Monitoring Training

Training logs are saved to the output directory. Key metrics to monitor:

- **Loss**: Should generally decrease over time
- **Evaluation loss**: Indicates generalization; if it increases while training loss decreases, you may be overfitting
- **Learning rate**: Follows the cosine schedule defined in the config

You can visualize training with TensorBoard:
```bash
tensorboard --logdir output/minicpmv45_invoice_lora
```

## Using Trained Adapters

### Option 1: Merge Adapter into Base Model

For production deployment, merge the LoRA adapter into the base model:

```bash
llamafactory-cli export \
  llamafactory_config.yaml \
  --adapter_name_or_path output/minicpmv45_invoice_lora/checkpoint-XXXX \
  --export_dir output/merged_model
```

Then update `MODEL_ADAPTOR` environment variable to point to `output/merged_model`.

### Option 2: Load Adapter Dynamically (Not Recommended for Production)

For testing, you can load the base model and adapter separately, but this adds latency.

## Best Practices

1. **Start small**: Train on a small dataset first to validate your pipeline
2. **Monitor overfitting**: Use the validation split (5% by default) to detect overfitting
3. **Save checkpoints frequently**: Set `save_steps` appropriately for your dataset size
4. **Experiment with learning rates**: If loss plateaus, try reducing the learning rate
5. **Use incremental training wisely**: Don't train on the same data repeatedly; add new examples
6. **Clean your data**: Remove low-quality or mislabeled examples for better results

## Troubleshooting

### CUDA Out of Memory

- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Reduce `cutoff_len` if your prompts are very long
- Enable DeepSpeed ZeRO optimization (advanced)

### Poor Convergence

- Check your data format matches ShareGPT exactly
- Ensure image paths are correct and images are readable
- Try increasing `num_train_epochs` or reducing `learning_rate`
- Validate that your training data is diverse and representative

### Adapter Not Loading

- Verify the checkpoint path is correct
- Ensure the checkpoint contains `adapter_config.json` and `adapter_model.safetensors`
- Check that you're using compatible versions of `peft` and `transformers`

## Advanced: DeepSpeed Integration

For multi-GPU training, you can enable DeepSpeed in the config:

```yaml
deepspeed: ds_config_zero2.json
```

This requires additional setup. See LLaMA-Factory documentation for details.

## References

- [LLaMA-Factory GitHub](https://github.com/hiyouga/LLaMA-Factory)
- [MiniCPM-V 4.5 Model Card](https://huggingface.co/openbmb/MiniCPM-V-4_5)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
