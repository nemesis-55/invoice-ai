# Model Upgrade Documentation: MiniCPM-V-2.6 → MiniCPM-V-4.5

**Date:** February 11, 2026  
**Author:** GitHub Copilot Agent  
**Task:** Upgrade from openbmb/MiniCPM-V-2_6 to openbmb/MiniCPM-V-4_5 and integrate LLamaFactory

---

## Table of Contents
1. [Overview](#overview)
2. [Key Changes in MiniCPM-V-4.5](#key-changes-in-minicpm-v-45)
3. [Migration Steps](#migration-steps)
4. [Files Modified](#files-modified)
5. [LLamaFactory Integration](#llamafactory-integration)
6. [Testing & Validation](#testing--validation)

---

## Overview

This document tracks the upgrade from MiniCPM-V-2.6 to MiniCPM-V-4.5 for the invoice-ai project. The upgrade includes:

- Model architecture update to leverage improved vision-language capabilities
- Integration with LLamaFactory for enhanced training workflows
- Dataset preparation updates to support new model requirements
- Configuration updates across training scripts

---

## Key Changes in MiniCPM-V-4.5

### Architecture Improvements
- **Base Models:** Qwen3-8B LLM + SigLIP2-400M vision encoder (8B total parameters)
- **Unified 3D-Resampler:** Revolutionary token compression (up to 96x for video)
  - 6 video frames (448x448) → only 64 tokens (vs 1,536 in typical MLLMs)
- **LLaVA-UHD Architecture:** High-res images (up to 1.8M pixels) with 4x fewer tokens
- **Hybrid Fast/Deep Thinking:** User-controllable reasoning modes

### Performance Benefits
- Superior OCR & document parsing (matches/exceeds GPT-4o, Gemini 2.x)
- 30+ languages support out-of-the-box
- ~47% less GPU memory, 9% of inference time vs larger alternatives
- State-of-the-art on OpenCompass (77.0+ score)

### Technical Differences from V-2.6
1. **LLM Type Change:** V-2.6 uses "minicpm" → V-4.5 uses "qwen3"
2. **Token Efficiency:** Much better compression for images/video
3. **Vision Encoder:** Upgraded to SigLIP2-400M
4. **Enhanced Capabilities:** Better document understanding for invoice extraction

---

## Migration Steps

### Step 1: Update Model References
Files to update:
- `training/finetune.py` - Default model path
- `training/finetune_ds.sh` - Shell script configurations
- `training/finetune_lora.sh` - LoRA training script
- `training/train.ipynb` - Jupyter notebook

Changes:
```python
# OLD
model_name_or_path: Optional[str] = field(default="openbmb/MiniCPM-V-2")

# NEW
model_name_or_path: Optional[str] = field(default="openbmb/MiniCPM-V-4_5")
```

### Step 2: Update LLM Type Configuration
The LLM backbone has changed from MiniCPM to Qwen3:

```bash
# OLD
LLM_TYPE="minicpm"

# NEW
LLM_TYPE="qwen3"
```

### Step 3: Dataset Preparation
The current dataset format should remain compatible, but we need to verify:
- Image format: MiniCPM-V-4.5 supports up to 1344x1344 high-resolution images
- Conversation format: Standard `<image>` tag-based format works
- Token limits: V-4.5 is more efficient, allowing more content per sequence

Current format (from `prepare_data/create_training_data.py`):
```json
{
  "id": "185486_1",
  "image": "path/to/image.jpg",
  "conversations": [
    {"role": "user", "content": "<image>\nExtract fields from the invoice..."},
    {"role": "assistant", "content": "{...extracted JSON...}"}
  ]
}
```

This format is compatible with MiniCPM-V-4.5 ✓

### Step 4: Training Configuration Updates
Update DeepSpeed configs if needed for new model architecture.

---

## Files Modified

### Core Training Files
1. **training/finetune.py**
   - Line 28: Updated default model to `openbmb/MiniCPM-V-4_5`
   - Line 53: Updated default LLM type to `qwen3`

2. **training/finetune_ds.sh**
   - Line 9: Updated MODEL variable
   - Line 15: Updated LLM_TYPE to `qwen3`
   - Comments updated to reflect V-4.5

3. **training/finetune_lora.sh**
   - Line 9: Updated MODEL variable
   - Line 14: Updated LLM_TYPE to `qwen3`
   - Comments updated to reflect V-4.5

4. **training/train.ipynb**
   - Model loading cells updated
   - References to V-2.6 replaced with V-4.5

### Documentation Files
5. **requirements.txt**
   - Added LLamaFactory dependencies (if needed)

6. **README.md**
   - Updated model information
   - Added LLamaFactory usage instructions

---

## LLamaFactory Integration

### Why LLamaFactory?
LLamaFactory provides:
- Unified interface for fine-tuning 100+ LLMs and VLMs
- Support for LoRA, QLoRA, and full fine-tuning
- Built-in support for MiniCPM-V models
- Web UI and CLI for easier training management
- Better monitoring and experiment tracking

### Installation
```bash
# Install LLamaFactory with MiniCPM-V support
pip install -e ".[torch,metrics,deepspeed,minicpm_v]"
```

### Dataset Format for LLamaFactory
LLamaFactory uses a similar but slightly different format:

```json
[
  {
    "messages": [
      {"content": "<image>Extract invoice fields...", "role": "user"},
      {"content": "{...extracted JSON...}", "role": "assistant"}
    ],
    "images": ["path/to/image.jpg"]
  }
]
```

### Configuration Files
Create `configs/minicpm_v45_invoice.yaml`:
```yaml
model_name_or_path: openbmb/MiniCPM-V-4_5
stage: sft
do_train: true
dataset: invoice_training
template: minicpm_v
cutoff_len: 8192
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16
output_dir: output/minicpm_v45_invoice
logging_steps: 10
save_steps: 1000
plot_loss: true
overwrite_output_dir: true
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 1.0e-6
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.01
bf16: true
ddp_timeout: 180000000
lora_rank: 64
lora_alpha: 64
lora_dropout: 0.05
lora_target: all
```

### Training with LLamaFactory
```bash
# Using CLI
llamafactory-cli train configs/minicpm_v45_invoice.yaml

# Using Python API
python -m llamafactory.train configs/minicpm_v45_invoice.yaml

# Using Web UI
llamafactory-cli webui
```

---

## Testing & Validation

### Pre-Migration Tests
- [ ] Backup current training data
- [ ] Document current model performance metrics
- [ ] Save current training configuration

### Post-Migration Tests
- [ ] Verify model loads correctly with new identifier
- [ ] Test dataset preparation pipeline
- [ ] Run small-scale training test (few steps)
- [ ] Validate inference with fine-tuned adapter
- [ ] Compare output quality with previous model

### Validation Checklist
- [ ] Model architecture matches V-4.5 specifications
- [ ] Training runs without errors
- [ ] Inference produces expected outputs
- [ ] Performance is equal or better than V-2.6
- [ ] LLamaFactory integration works correctly

---

## Notes and Observations

### Compatibility Notes
- Handler.py uses `MODEL_ADAPTOR` env variable - should work with new base model
- Current dataset format is compatible with V-4.5
- DeepSpeed configurations (ZeRO2/ZeRO3) should work without changes
- LoRA configurations may need minor adjustments

### Performance Expectations
- Faster training due to better token efficiency
- Better document understanding for invoice extraction
- Potentially higher accuracy on OCR tasks
- Lower memory footprint during training

### Future Improvements
1. Experiment with V-4.5's fast/deep thinking modes
2. Leverage improved video understanding if needed
3. Test multilingual invoice support
4. Optimize for quantized inference (int4/AWQ)

---

## References

1. [MiniCPM-V-4.5 Model Card](https://huggingface.co/openbmb/MiniCPM-V-4_5)
2. [MiniCPM-V-4.5 Technical Report](https://arxiv.org/html/2509.18154v1)
3. [LLamaFactory Documentation](https://github.com/hiyouga/LLaMA-Factory)
4. [MiniCPM-V Cookbook](https://minicpm-o.readthedocs.io/)
5. [LLamaFactory Fine-tuning Guide](https://minicpm-o.readthedocs.io/en/latest/finetune/llamafactory.html)

---

## Change Log

### [2026-02-11] - Initial Planning
- Created upgrade documentation
- Researched MiniCPM-V-4.5 architecture
- Planned LLamaFactory integration approach
- Identified files requiring updates

### [2026-02-11] - Implementation Complete
- ✅ Updated all model references from V-2.6 to V-4.5
- ✅ Changed LLM type from "minicpm" to "qwen3"
- ✅ Created LLamaFactory configuration files
- ✅ Added training scripts for both native and LLamaFactory workflows
- ✅ Created comprehensive documentation (UPGRADE_NOTES.md, LLAMAFACTORY_GUIDE.md, MIGRATION_CHECKLIST.md)
- ✅ Updated README.md with new information
- ✅ Added validation script (validate_config.py)
- ✅ Verified dataset format compatibility
- ✅ Made all training scripts executable
- ✅ Updated requirements.txt

---

*This document will be updated as the migration progresses.*
