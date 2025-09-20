# Parameter Adjustment Guide

This guide explains how to adjust parameters for optimal performance based on your GPU and requirements.

## Quick Setup

1. **Edit the GPU Profile**: Open `gpu_config.json` and change the `active_profile` to match your GPU:
   ```json
   "active_profile": "rtx_4090"  // or "rtx_5090"
   ```

2. **Restart the application** to apply the new settings.

## GPU Profiles

### RTX 4090 Profile
- **VRAM**: 24GB
- **Precision**: 8-bit quantization (balanced performance/memory)
- **Max Tokens**: 4096 (suitable for most invoice processing)
- **Use Case**: Production inference with memory efficiency

### RTX 5090 Profile  
- **VRAM**: 32GB
- **Precision**: 16-bit (highest quality)
- **Max Tokens**: 8192 (extended context for complex documents)
- **Use Case**: Maximum quality inference, complex document processing

## Manual Parameter Adjustment

If you need custom settings, you can modify the `gpu_config.json` file:

### Model Precision Options
```json
"model_precision": "16bit"  // Highest quality, most VRAM usage
"model_precision": "8bit"   // Balanced quality and memory
"model_precision": "4bit"   // Lowest memory usage, reduced quality
```

### GPU Device Options
```json
"gpu_device": "single"      // Use first GPU only (recommended)
"gpu_device": "auto"        // Distribute across all GPUs
"gpu_device": "cuda:0"      // Use specific GPU (0, 1, 2, etc.)
```

### Token Limits
```json
"max_new_tokens": 4096      // Standard for most invoices
"max_new_tokens": 8192      // For complex/multi-page documents
"max_new_tokens": 2048      // For simple invoices (faster processing)
```

## Memory Usage Guidelines

| Precision | RTX 4090 (24GB) | RTX 5090 (32GB) | Quality | Speed |
|-----------|------------------|------------------|---------|-------|
| 16bit     | ⚠️ Tight fit    | ✅ Comfortable   | Highest | Slow  |
| 8bit      | ✅ Recommended  | ✅ Efficient     | High    | Fast  |
| 4bit      | ✅ Conservative | ✅ Very efficient| Good    | Fastest|

## Performance Tuning

### For Maximum Quality
```json
{
  "model_precision": "16bit",
  "max_new_tokens": 8192,
  "gpu_device": "single"
}
```

### For Maximum Speed
```json
{
  "model_precision": "4bit",
  "max_new_tokens": 2048,
  "gpu_device": "single"
}
```

### For Balanced Performance
```json
{
  "model_precision": "8bit",
  "max_new_tokens": 4096,
  "gpu_device": "single"
}
```

## Troubleshooting

### Out of Memory Errors
1. Switch to lower precision: `16bit` → `8bit` → `4bit`
2. Reduce max tokens: `8192` → `4096` → `2048`
3. Ensure no other GPU processes are running

### Poor Quality Results
1. Increase precision: `4bit` → `8bit` → `16bit`
2. Increase max tokens for complex documents
3. Verify the model is using the correct GPU

### Slow Performance
1. Use `4bit` or `8bit` precision
2. Use `single` GPU device mode
3. Reduce max tokens if processing simple documents

## Environment Variables Override

You can still use environment variables to override settings:
```bash
MODEL_PRECISION=8bit GPU_DEVICE=cuda:1 python handler.py
```

These will take precedence over the GPU config file settings.

## Advanced Configuration

For custom GPU configurations not covered by the profiles, modify the `fallback_settings` in `gpu_config.json` or create a new profile section.

## Validation

Run the test script to validate your configuration:
```bash
python test_config.py
```

This guide is read-only. Modify `gpu_config.json` to change settings.