# MiniCPM-V 4.5 Upgrade Guide

This document describes the upgrade from MiniCPM-V 2.6 to MiniCPM-V 4.5 and all associated changes.

## What's New

### 🚀 MiniCPM-V 4.5 Model
- **Model ID**: `openbmb/MiniCPM-V-4_5`
- **Architecture**: Qwen3-8B LLM + SigLIP2-400M vision encoder (8B params total)
- **Precision**: bfloat16 (previously float16) for better quality
- **Features**: Hybrid fast/deep thinking modes

### 🧠 Deep Thinking Mode
Enable step-by-step reasoning for complex tasks:

```python
# Environment variable (global default)
export DEEP_THINKING=true

# Per-request override
payload = {
    "action": "INVOICE_EXTRACTION",
    "data": {
        "pdf_data": "...",
        "deep_thinking": true  # Override for this request
    }
}
```

When enabled:
- Prepends system message: "You are a helpful assistant. Think step by step carefully before responding."
- Increases `max_new_tokens` from 512 to 2048
- Better accuracy for complex invoice extraction

### 🎓 LLaMA-Factory Training
New incremental LoRA training support:

```bash
cd training

# Initial training
./run_incremental_train.sh

# Continue from checkpoint
./run_incremental_train.sh output/minicpmv45_invoice_lora/checkpoint-1000
```

See `training/README.md` for complete documentation.

## Migration Guide

### Environment Variables

**New variables:**
```bash
DEEP_THINKING=false  # Enable deep thinking mode (default: false)
```

**Updated defaults:**
```bash
MODEL_ADAPTOR=openbmb/MiniCPM-V-4_5  # Changed from GothiaDigitalSolutions/invoice-extractor-3.0
```

### Docker Deployment

**Update your deployment:**
```bash
# Pull new image (will be rebuilt with updated Dockerfile)
docker pull your-registry/invoice-ai:latest

# Or rebuild locally
docker build -t invoice-ai .
```

**New base image:**
- Old: `runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04`
- New: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`

### API Changes

**All payload models now support `deep_thinking`:**

```python
# Invoice extraction with deep thinking
{
    "action": "INVOICE_EXTRACTION",
    "data": {
        "pdf_data": "base64_encoded_pdf",
        "page_number": "0",
        "deep_thinking": true  # NEW: Optional per-request override
    }
}

# Prompt with deep thinking
{
    "action": "PROMPT",
    "data": {
        "prompt": "Extract data...",
        "deep_thinking": true  # NEW: Optional per-request override
    }
}

# Assistant with deep thinking
{
    "action": "ASSISTANT",
    "data": {
        "prompt": "Analyze this invoice...",
        "attachments": [...],
        "deep_thinking": true  # NEW: Optional per-request override
    }
}

# Classification with deep thinking
{
    "action": "CLASSIFICATION",
    "data": {
        "prompt": "Classify...",
        "image": "base64_image",
        "deep_thinking": true  # NEW: Optional per-request override
    }
}
```

### Error Handling Improvements

**More consistent error responses:**
- All handlers now return `{"error": "message"}` instead of raising exceptions
- Better JSON parsing error handling
- Unknown actions return proper error messages

## Bug Fixes

1. ✅ Removed duplicate `base64` import
2. ✅ Removed duplicate `fitz` import
3. ✅ Removed unused `from peft import PeftModel`
4. ✅ Fixed `handle_classification` error handling
5. ✅ Fixed `run()` to handle unknown actions
6. ✅ Fixed JSON parsing crashes in multiple handlers

## Performance Considerations

### Memory Usage
- MiniCPM-V 4.5 has same size (8B params) but uses bfloat16
- Deep thinking mode uses ~4x more tokens (2048 vs 512)
- Consider GPU memory when enabling deep thinking

### Inference Speed
- Standard mode: Same speed as before
- Deep thinking mode: Slower due to increased token generation
- Use deep thinking only when needed for complex tasks

## Training Your Own Model

1. **Prepare data** in ShareGPT format (see `training/README.md`)
2. **Update config** if needed (`training/llamafactory_config.yaml`)
3. **Run training**: `cd training && ./run_incremental_train.sh`
4. **Resume from checkpoint**: `./run_incremental_train.sh path/to/checkpoint`
5. **Deploy**: Update `MODEL_ADAPTOR` to your trained model path

## Rollback Plan

If you need to rollback to MiniCPM-V 2.6:

```bash
# Revert the changes
git revert <commit-hash>

# Or manually update environment variables
export MODEL_ADAPTOR=GothiaDigitalSolutions/invoice-extractor-3.0

# And redeploy
```

## Testing

Verify the upgrade works:

```python
import requests
import base64

# Test with deep thinking disabled
response = requests.post(
    "https://your-endpoint/run",
    json={
        "input": {
            "action": "PROMPT",
            "data": {
                "prompt": "Say hello",
                "deep_thinking": false
            }
        }
    }
)

# Test with deep thinking enabled
response = requests.post(
    "https://your-endpoint/run",
    json={
        "input": {
            "action": "PROMPT",
            "data": {
                "prompt": "Solve this complex problem step by step...",
                "deep_thinking": true
            }
        }
    }
)
```

## Support

For issues or questions:
- Check `training/README.md` for training documentation
- Review the [MiniCPM-V 4.5 model card](https://huggingface.co/openbmb/MiniCPM-V-4_5)
- Check handler.py for implementation details

## Changelog

### v4.5.0 - MiniCPM-V 4.5 Upgrade
- ✨ Upgraded to MiniCPM-V 4.5 with Qwen3-8B backbone
- ✨ Added Deep Thinking mode support
- ✨ Added LLaMA-Factory incremental LoRA training
- 🐛 Fixed 6 bugs in handler.py
- 📦 Updated dependencies (PyTorch 2.4.0, CUDA 12.4.1)
- 📚 Comprehensive training documentation
