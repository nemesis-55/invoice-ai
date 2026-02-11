# Migration Checklist: MiniCPM-V-2.6 → MiniCPM-V-4.5

Use this checklist to ensure a smooth migration to MiniCPM-V-4.5 with LLamaFactory support.

## Pre-Migration

- [ ] **Backup Current Setup**
  - [ ] Save current training data
  - [ ] Save current model weights/adapters
  - [ ] Document current hyperparameters
  - [ ] Save training logs and metrics

- [ ] **Environment Preparation**
  - [ ] Python 3.10+ installed
  - [ ] CUDA 11.8+ or 12.1+ installed
  - [ ] Sufficient GPU memory (24GB+ for LoRA, 80GB+ for full)
  - [ ] Disk space for model weights (~16GB for base model)

## Code Migration

- [x] **Model References Updated**
  - [x] `training/finetune.py` - Model path changed to `openbmb/MiniCPM-V-4_5`
  - [x] `training/finetune.py` - LLM type changed to `qwen3`
  - [x] `training/finetune_ds.sh` - Updated model and LLM type
  - [x] `training/finetune_lora.sh` - Updated model and LLM type

- [x] **LLamaFactory Integration**
  - [x] Created `llamafactory_configs/` directory
  - [x] Created `dataset_info.json` for dataset registration
  - [x] Created `minicpm_v45_lora.yaml` configuration
  - [x] Created `minicpm_v45_full.yaml` configuration
  - [x] Created training scripts for LLamaFactory

- [x] **Documentation**
  - [x] Created `UPGRADE_NOTES.md`
  - [x] Created `LLAMAFACTORY_GUIDE.md`
  - [x] Updated `README.md`
  - [x] Created validation script

## Installation & Setup

- [ ] **Update Python Dependencies**
  ```bash
  pip install --upgrade transformers>=4.44.0
  pip install --upgrade torch>=2.1.0
  pip install --upgrade peft>=0.16.0
  ```

- [ ] **Install LLamaFactory (Optional but Recommended)**
  ```bash
  pip install llamafactory[torch,metrics,deepspeed,minicpm_v]
  ```

- [ ] **Verify Installation**
  ```bash
  python validate_config.py
  ```

## Dataset Preparation

- [ ] **Prepare Training Data**
  - [ ] Set environment variables:
    ```bash
    export RAW_DATA_OUTPUT="path/to/raw_data.json"
    export TRAIN_DATA_PATH="data/train_data.json"
    export TEST_DATA_PATH="data/test_data.json"
    export SPLIT_RATIO="0.9"
    ```
  - [ ] Run data preparation:
    ```bash
    python prepare_data/create_training_data.py
    ```
  - [ ] Verify data format with validation script

- [ ] **Validate Dataset**
  - [ ] Check all images exist and are readable
  - [ ] Verify JSON format is correct
  - [ ] Confirm train/test split ratio
  - [ ] Review sample conversations

## Testing & Validation

- [ ] **Test Native Training Scripts**
  - [ ] Test LoRA script runs without errors:
    ```bash
    # Dry run or short test
    bash training/finetune_lora.sh
    ```
  - [ ] Test DeepSpeed script configuration:
    ```bash
    bash training/finetune_ds.sh
    ```

- [ ] **Test LLamaFactory Integration**
  - [ ] Copy dataset_info.json to LLamaFactory:
    ```bash
    cp llamafactory_configs/dataset_info.json \
       $(python -c "import llamafactory, os; print(os.path.join(os.path.dirname(llamafactory.__file__), 'data'))")/
    ```
  - [ ] Test LoRA training:
    ```bash
    bash training/train_llamafactory_lora.sh
    ```
  - [ ] Test Web UI:
    ```bash
    llamafactory-cli webui
    ```

- [ ] **Smoke Test Training**
  - [ ] Run training for 10-50 steps
  - [ ] Verify loss is decreasing
  - [ ] Check GPU memory usage
  - [ ] Confirm checkpoints are saved
  - [ ] Validate tensorboard logs

## Model Inference Testing

- [ ] **Test Base Model Loading**
  ```python
  from transformers import AutoModel, AutoTokenizer
  
  model = AutoModel.from_pretrained(
      "openbmb/MiniCPM-V-4_5",
      trust_remote_code=True,
      torch_dtype=torch.float16,
      device_map="auto"
  )
  tokenizer = AutoTokenizer.from_pretrained(
      "openbmb/MiniCPM-V-4_5",
      trust_remote_code=True
  )
  ```

- [ ] **Test Fine-tuned Model Loading**
  ```python
  from peft import PeftModel
  
  # Load with LoRA adapter
  model = PeftModel.from_pretrained(
      model,
      "output/minicpm_v45_lora_invoice"
  )
  ```

- [ ] **Test Inference**
  - [ ] Load a sample invoice image
  - [ ] Run extraction with the model
  - [ ] Verify output quality
  - [ ] Compare with previous model (if available)

## Production Deployment

- [ ] **Update Deployment Configuration**
  - [ ] Update `handler.py` MODEL_ADAPTOR environment variable
  - [ ] Update Docker images/containers
  - [ ] Update API endpoints if needed
  - [ ] Test deployment in staging environment

- [ ] **Performance Validation**
  - [ ] Benchmark inference speed
  - [ ] Test with various invoice types
  - [ ] Validate extraction accuracy
  - [ ] Monitor GPU memory usage
  - [ ] Check for any regressions

## Documentation & Handoff

- [ ] **Update Internal Documentation**
  - [ ] Document new training procedures
  - [ ] Update deployment guides
  - [ ] Document troubleshooting steps
  - [ ] Share migration checklist with team

- [ ] **Knowledge Transfer**
  - [ ] Train team on LLamaFactory usage
  - [ ] Demonstrate Web UI features
  - [ ] Review configuration options
  - [ ] Share best practices

## Post-Migration Monitoring

- [ ] **Week 1: Close Monitoring**
  - [ ] Monitor extraction accuracy daily
  - [ ] Track inference latency
  - [ ] Check for errors/exceptions
  - [ ] Collect user feedback

- [ ] **Week 2-4: Regular Checks**
  - [ ] Review weekly metrics
  - [ ] Fine-tune hyperparameters if needed
  - [ ] Address any issues found
  - [ ] Document learnings

## Rollback Plan (If Needed)

In case of critical issues:

- [ ] **Immediate Rollback**
  - [ ] Revert to previous model version
  - [ ] Restore previous training scripts
  - [ ] Notify stakeholders
  - [ ] Document issues encountered

- [ ] **Investigation**
  - [ ] Analyze failure points
  - [ ] Review logs and errors
  - [ ] Test fixes in isolation
  - [ ] Plan re-migration

---

## Quick Reference Commands

### Validation
```bash
python validate_config.py
```

### Training (Choose One)
```bash
# Native LoRA
bash training/finetune_lora.sh

# Native Full (DeepSpeed)
bash training/finetune_ds.sh

# LLamaFactory LoRA
bash training/train_llamafactory_lora.sh

# LLamaFactory Full
bash training/train_llamafactory_full.sh

# LLamaFactory Web UI
llamafactory-cli webui
```

### Monitoring
```bash
# TensorBoard
tensorboard --logdir output/[model_output_dir]/logs

# LlamaBoard
llamafactory-cli board
```

---

## Support Resources

- **Documentation:**
  - `UPGRADE_NOTES.md` - Detailed technical changes
  - `LLAMAFACTORY_GUIDE.md` - Complete LLamaFactory guide
  - `README.md` - Quick start and overview

- **External Resources:**
  - [MiniCPM-V-4.5 Model Card](https://huggingface.co/openbmb/MiniCPM-V-4_5)
  - [LLamaFactory GitHub](https://github.com/hiyouga/LLaMA-Factory)
  - [MiniCPM-V Documentation](https://minicpm-o.readthedocs.io/)

---

**Completion Date:** __________  
**Migrated By:** __________  
**Notes:** ___________________________________
