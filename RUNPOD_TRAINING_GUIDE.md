# RunPod Training Guide for MiniCPM-V-4.5

**Complete guide for training MiniCPM-V-4.5 on RunPod GPU infrastructure**

---

## Table of Contents
1. [GPU Configuration Recommendations](#gpu-configuration-recommendations)
2. [Getting Started with RunPod](#getting-started-with-runpod)
3. [Setting Up Your Training Pod](#setting-up-your-training-pod)
4. [Training Workflow](#training-workflow)
5. [Cost Optimization](#cost-optimization)
6. [Troubleshooting](#troubleshooting)

---

## GPU Configuration Recommendations

### Understanding MiniCPM-V-4.5 Requirements

**Model Specifications:**
- **Parameters:** 8B (Qwen3-8B LLM + SigLIP2-400M vision encoder)
- **Base Memory:** ~16GB for model weights (FP16)
- **Training Memory:** Additional for gradients, optimizer states, activations

### Recommended GPU Configurations

#### 🥉 Budget Option: LoRA Fine-tuning (Single GPU)

**GPU Options:**
- **RTX 4090** (24GB VRAM)
- **RTX A6000** (48GB VRAM) - Recommended
- **A40** (48GB VRAM)

**Configuration:**
```yaml
Training Method: LoRA
Batch Size: 1-2
Gradient Accumulation: 4-8
VRAM Usage: ~20-24GB
Training Speed: ~2-3 hours per 1000 steps
Cost: $0.30-0.50/hour (spot pricing)
```

**Best For:**
- Domain adaptation
- Quick iterations
- Budget-conscious projects
- Single-user experiments

---

#### 🥈 Recommended: LoRA Multi-GPU (2x GPUs)

**GPU Options:**
- **2x RTX 4090** (24GB each = 48GB total)
- **2x A40** (48GB each = 96GB total) - Recommended
- **2x L40** (48GB each = 96GB total)

**Configuration:**
```yaml
Training Method: LoRA
Batch Size: 2-4
Gradient Accumulation: 4
VRAM Usage: ~40-48GB total
Training Speed: ~1-1.5 hours per 1000 steps
Cost: $0.60-1.00/hour (spot pricing)
```

**Best For:**
- Production fine-tuning
- Faster iteration cycles
- Larger batch sizes
- Better generalization

---

#### 🥇 Premium: Full Fine-tuning (Multi-GPU)

**GPU Options:**
- **2x A100 80GB** (160GB total) - Recommended
- **4x A100 40GB** (160GB total)
- **4x A40 48GB** (192GB total)
- **2x H100 80GB** (160GB total) - Best performance

**Configuration:**
```yaml
Training Method: Full Fine-tuning + DeepSpeed ZeRO-2
Batch Size: 4-8
Gradient Accumulation: 2-4
VRAM Usage: ~120-160GB total
Training Speed: ~0.5-1 hour per 1000 steps
Cost: $2.00-4.00/hour (spot pricing)
```

**Best For:**
- Maximum model adaptation
- Large-scale training
- Research projects
- Production deployments requiring best accuracy

---

#### 💎 Enterprise: Large-Scale Training

**GPU Options:**
- **4x A100 80GB** (320GB total)
- **8x A100 40GB** (320GB total)
- **4x H100 80GB** (320GB total) - Best performance

**Configuration:**
```yaml
Training Method: Full Fine-tuning + DeepSpeed ZeRO-3
Batch Size: 8-16
Gradient Accumulation: 2
VRAM Usage: ~200-300GB total
Training Speed: ~15-30 min per 1000 steps
Cost: $4.00-8.00/hour (spot pricing)
```

**Best For:**
- Continuous pre-training
- Multi-task fine-tuning
- Extremely large datasets
- Research institutions

---

### Quick Comparison Table

| Configuration | GPUs | VRAM | Cost/hr (Spot) | Training Speed | Use Case |
|--------------|------|------|----------------|----------------|----------|
| **Budget** | 1x RTX 4090 | 24GB | $0.30-0.50 | Slow | Experiments |
| **Recommended** | 2x A40 | 96GB | $0.60-1.00 | Medium | Production LoRA |
| **Premium** | 2x A100 80GB | 160GB | $2.00-4.00 | Fast | Full Fine-tuning |
| **Enterprise** | 4x A100 80GB | 320GB | $4.00-8.00 | Very Fast | Large-scale |

**Note:** Prices are approximate and vary based on availability. Spot pricing can be 50-80% cheaper than on-demand.

---

## Getting Started with RunPod

### 1. Create RunPod Account

1. Visit [RunPod.io](https://runpod.io)
2. Sign up for an account
3. Add payment method
4. Add credits ($10-100 recommended for initial testing)

### 2. Understanding Pod Types

**Secure Cloud Pods:**
- Stable, guaranteed availability
- Best for production training
- Higher cost but no interruptions

**Community Cloud (Spot) Pods:**
- 50-80% cost savings
- Can be interrupted
- Best for checkpointed training
- **Recommended** for cost-effective training

### 3. API Keys (Optional)

For programmatic access:
1. Go to Settings → API Keys
2. Generate new API key
3. Save securely (needed for automation)

---

## Setting Up Your Training Pod

### Step 1: Select GPU Configuration

1. Navigate to **"Pods"** → **"GPU Pods"**
2. Choose deployment type:
   - **Community Cloud** (Recommended for training)
   - **Secure Cloud** (If you need guaranteed availability)
3. Filter by GPU type (see recommendations above)
4. Sort by price to find best value

### Step 2: Configure Pod Settings

**Container Image Options:**

**Option A: Use PyTorch Base Image (Recommended)**
```
runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04
```

**Option B: Use Custom Dockerfile**
- Upload your Dockerfile to GitHub
- RunPod will build it automatically

### Step 3: Set Up Network Volumes

**Why Network Volumes?**
- Persist data across pod restarts
- Share datasets between pods
- Save checkpoints safely
- Don't lose work on spot interruptions

**Creating Network Volume:**
1. Go to **"Storage"** → **"Network Volumes"**
2. Click **"Create Network Volume"**
3. Configure:
   - Name: `invoice-training-data`
   - Size: 100GB-500GB (based on dataset size)
   - Region: Same as your pod
4. Click **"Create"**

**Mounting in Pod:**
- Mount path: `/workspace/data` (recommended)
- This will be accessible from your training scripts

### Step 4: Environment Variables

Set these environment variables in pod configuration:

```bash
# Hugging Face (for model downloads)
HF_TOKEN=your_huggingface_token

# Data paths
RAW_DATA_OUTPUT=/workspace/data/raw_data.json
TRAIN_DATA_PATH=/workspace/data/train_data.json

# Training output
OUTPUT_DIR=/workspace/data/output

# Wandb (optional, for logging)
WANDB_API_KEY=your_wandb_key
WANDB_PROJECT=invoice-ai-training
```

### Step 5: Deploy Pod

1. Review configuration
2. Check estimated cost
3. Click **"Deploy"** or **"Deploy On-Demand"**
4. Wait for pod to start (usually 1-3 minutes)

---

## Training Workflow

### Initial Setup (First Time Only)

Once your pod is running, connect via SSH or Web Terminal:

```bash
# 1. Clone repository
cd /workspace
git clone https://github.com/Arindam2002/invoice-ai.git
cd invoice-ai

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Optional: Install LLamaFactory
pip install llamafactory[torch,metrics,deepspeed,minicpm_v]

# 3. Validate configuration
python validate_config.py
```

### Upload Training Data

**Option A: Using RunPod's File Browser**
1. Open pod's web terminal
2. Use file browser to upload dataset
3. Save to mounted network volume

**Option B: Using rclone/rsync**
```bash
# From your local machine
rsync -avz -e "ssh -p PORT" \
  ./data/ \
  root@POD_ID.runpod.io:/workspace/data/
```

**Option C: Download from Cloud Storage**
```bash
# From Azure Blob Storage
pip install azure-storage-blob
python scripts/download_data_from_azure.py

# From AWS S3
aws s3 sync s3://your-bucket/data/ /workspace/data/

# From Google Cloud Storage
gsutil -m rsync -r gs://your-bucket/data/ /workspace/data/
```

### Prepare Dataset

```bash
# Set environment variables
export RAW_DATA_OUTPUT="/workspace/data/raw_data.json"
export TRAIN_DATA_PATH="/workspace/data/train_data.json"

# Run data preparation
cd /workspace/invoice-ai
python prepare_data/create_training_data.py

# Verify data was created
ls -lh /workspace/data/
```

**Note:** The script creates a single `train_data.json` file. LLamaFactory automatically handles train/validation splitting during training based on the `val_size` parameter in the config files (default: 0.1).

### Start Training

#### Method 1: LoRA Training (Recommended for Most Users)

```bash
cd /workspace/invoice-ai

# Edit configuration if needed
nano llamafactory_configs/minicpm_v45_lora.yaml

# Update output path to network volume
# output_dir: /workspace/data/output/minicpm_v45_lora_invoice

# Start training
bash training/train_llamafactory_lora.sh
```

#### Method 2: Full Fine-tuning (Multi-GPU)

```bash
cd /workspace/invoice-ai

# Check GPU count
nvidia-smi

# Edit configuration
nano llamafactory_configs/minicpm_v45_full.yaml

# Update paths to network volume
# output_dir: /workspace/data/output/minicpm_v45_full_invoice
# dataset_dir: /workspace/data

# Start training with DeepSpeed
bash training/train_llamafactory_full.sh
```

#### Method 3: Native Scripts (Advanced)

```bash
cd /workspace/invoice-ai/training

# For LoRA
bash finetune_lora.sh

# For Full Fine-tuning with DeepSpeed
bash finetune_ds.sh
```

### Monitor Training

**Option A: TensorBoard**
```bash
# In a separate terminal/tmux session
tensorboard --logdir /workspace/data/output/[model_dir]/logs --host 0.0.0.0 --port 6006
```

Access via: `http://POD_ID.runpod.io:6006`

**Option B: Weights & Biases (Wandb)**
```bash
# Already configured via environment variables
# View at: https://wandb.ai/your-username/invoice-ai-training
```

**Option C: Watch Logs**
```bash
# Watch training progress
tail -f /workspace/data/output/[model_dir]/train.log

# Or use tmux to keep multiple views
tmux new -s training
# Ctrl+B, then D to detach
# tmux attach -t training to reattach
```

### Checkpoint Management

**Automatic Checkpoints:**
Training scripts save checkpoints automatically to:
```
/workspace/data/output/minicpm_v45_lora_invoice/checkpoint-{step}/
```

**Manual Backup:**
```bash
# Compress checkpoints
cd /workspace/data/output
tar -czf checkpoint-5000.tar.gz minicpm_v45_lora_invoice/checkpoint-5000/

# Upload to cloud storage (recommended)
# To Azure
az storage blob upload \
  --account-name YOUR_ACCOUNT \
  --container-name checkpoints \
  --file checkpoint-5000.tar.gz \
  --name checkpoint-5000.tar.gz

# To S3
aws s3 cp checkpoint-5000.tar.gz s3://your-bucket/checkpoints/

# To Google Cloud
gsutil cp checkpoint-5000.tar.gz gs://your-bucket/checkpoints/
```

---

## Cost Optimization

### 1. Use Spot/Community Cloud Pods

**Savings:** 50-80% vs on-demand
**Trade-off:** May be interrupted
**Solution:** Frequent checkpointing (every 500-1000 steps)

### 2. Right-Size Your GPU

**Don't over-provision:**
- Use `nvidia-smi` to monitor actual VRAM usage
- If using <70% VRAM, consider smaller GPU
- If at >95% VRAM, consider larger or additional GPUs

### 3. Checkpoint Frequently

```yaml
# In your training config
save_steps: 500  # Save every 500 steps
save_total_limit: 3  # Keep only last 3 checkpoints
```

**Benefits:**
- Minimize lost work on spot interruptions
- Allows resuming from any point
- Disk space management

### 4. Use Mixed Precision Training

Already enabled in configs:
```yaml
bf16: true  # or fp16: true
```

**Benefits:**
- 2x faster training
- 40-50% less VRAM usage
- Minimal accuracy impact

### 5. Optimize Batch Size & Gradient Accumulation

```yaml
per_device_train_batch_size: 1  # Fit in VRAM
gradient_accumulation_steps: 8   # Effective batch = 1 × 8 = 8
```

Find the sweet spot:
1. Start with batch_size=1, grad_accum=8
2. Monitor GPU utilization
3. If <80%, increase batch_size
4. Adjust grad_accum to maintain effective batch size

### 6. Terminate Idle Pods

**Important:** Always stop pods when not in use!
- Training finished? **Stop the pod immediately**
- Taking a break? **Stop the pod**
- Debugging code? Use cheaper CPU pod

### 7. Use Persistent Volumes Wisely

**Network Volumes:**
- Charged per GB per hour
- Keep only essential data
- Delete old checkpoints regularly

```bash
# Clean old checkpoints
cd /workspace/data/output/minicpm_v45_lora_invoice
ls -t | tail -n +4 | xargs rm -rf  # Keep only 3 latest
```

---

## Troubleshooting

### Problem: Pod Won't Start

**Cause:** GPU not available in selected region

**Solution:**
1. Try different region
2. Switch to different GPU type
3. Use on-demand instead of spot
4. Wait 10-30 minutes and retry

### Problem: Out of Memory (OOM)

**Symptoms:**
```
CUDA out of memory. Tried to allocate X GB
RuntimeError: CUDA error: out of memory
```

**Solutions:**

1. **Reduce batch size:**
   ```yaml
   per_device_train_batch_size: 1
   ```

2. **Increase gradient accumulation:**
   ```yaml
   gradient_accumulation_steps: 16
   ```

3. **Enable gradient checkpointing:**
   ```yaml
   gradient_checkpointing: true
   ```

4. **Use smaller LoRA rank:**
   ```yaml
   lora_rank: 32  # Instead of 64
   ```

5. **Use DeepSpeed ZeRO-3:**
   ```yaml
   deepspeed: training/ds_config_zero3.json
   ```

6. **Upgrade to larger GPU**

### Problem: Training Very Slow

**Check GPU Utilization:**
```bash
watch -n 1 nvidia-smi
```

**Solutions:**

1. **I/O bottleneck:**
   - Ensure data is on fast SSD/network volume
   - Increase data loader workers
   ```yaml
   dataloader_num_workers: 4
   ```

2. **Small batch size:**
   - Increase batch size if VRAM allows
   - Use multiple GPUs

3. **CPU bottleneck:**
   - Check CPU usage with `htop`
   - Reduce preprocessing workers

### Problem: Pod Interrupted (Spot Instance)

**Symptoms:** Training stops, pod terminates

**Prevention:**
- Save checkpoints frequently (every 500-1000 steps)
- Use network volumes for all important data

**Recovery:**
1. Deploy new pod with same configuration
2. Mount same network volume
3. Resume training from last checkpoint:
   ```bash
   cd /workspace/invoice-ai
   # Training scripts auto-detect and resume from checkpoint
   bash training/train_llamafactory_lora.sh
   ```

### Problem: Connection Lost

**Solutions:**

1. **Use tmux/screen:**
   ```bash
   tmux new -s training
   # Run training command
   # Ctrl+B, then D to detach
   # Reconnect anytime with: tmux attach -t training
   ```

2. **Run in background:**
   ```bash
   nohup bash training/train_llamafactory_lora.sh > train.log 2>&1 &
   ```

3. **Monitor via logs:**
   ```bash
   tail -f train.log
   ```

### Problem: Network Volume Not Accessible

**Check mount:**
```bash
df -h | grep workspace
ls -la /workspace/data/
```

**Solutions:**
1. Ensure volume is in same region as pod
2. Remount volume in pod settings
3. Check volume status in RunPod dashboard

### Problem: Model Download Fails

**Symptoms:**
```
Cannot connect to huggingface.co
Connection timeout
```

**Solutions:**

1. **Check HF_TOKEN:**
   ```bash
   echo $HF_TOKEN
   huggingface-cli login --token $HF_TOKEN
   ```

2. **Pre-download model:**
   ```bash
   python -c "
   from transformers import AutoModel
   AutoModel.from_pretrained('openbmb/MiniCPM-V-4_5', trust_remote_code=True)
   "
   ```

3. **Use model cache on network volume:**
   ```bash
   export HF_HOME=/workspace/data/hf_cache
   ```

---

## Advanced Tips

### Multi-Node Training

For very large-scale training across multiple pods:

1. Set up pods in same region
2. Configure network communication
3. Use DeepSpeed with multi-node config
4. Requires more advanced setup (contact RunPod support)

### Jupyter Notebook Setup

```bash
# Install Jupyter
pip install jupyter jupyterlab

# Start Jupyter on all interfaces
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Access via: http://POD_ID.runpod.io:8888
```

### VS Code Remote Connection

1. Install "Remote - SSH" extension in VS Code
2. Get SSH command from RunPod pod details
3. Add to VS Code SSH config
4. Connect and code directly on pod

### Automation with RunPod API

```python
import runpod

# Initialize
runpod.api_key = "your_api_key"

# Start pod programmatically
pod = runpod.create_pod(
    name="invoice-training",
    image_name="runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel",
    gpu_type_id="NVIDIA A40",
    cloud_type="COMMUNITY",
    volume_id="your_volume_id",
    env={
        "HF_TOKEN": "your_token",
        "TRAIN_DATA_PATH": "/workspace/data/train_data.json"
    }
)

print(f"Pod ID: {pod['id']}")
```

---

## Cost Estimation Calculator

### Example Training Scenarios

**Scenario 1: Quick LoRA Experiment (RTX 4090)**
- Duration: 2 hours
- GPU: 1x RTX 4090 (spot)
- Cost: $0.40/hour × 2 = **$0.80**

**Scenario 2: Production LoRA Training (2x A40)**
- Duration: 8 hours  
- GPU: 2x A40 (spot)
- Cost: $0.80/hour × 8 = **$6.40**

**Scenario 3: Full Fine-tuning (2x A100 80GB)**
- Duration: 12 hours
- GPU: 2x A100 80GB (spot)
- Cost: $2.50/hour × 12 = **$30.00**

**Scenario 4: Large-Scale Training (4x A100 80GB)**
- Duration: 24 hours
- GPU: 4x A100 80GB (on-demand)
- Cost: $5.00/hour × 24 = **$120.00**

**Network Volume:**
- 200GB volume: ~$0.10/hour
- 24 hours = **$2.40**

**Total Estimated Costs:**
- Quick experiment: **~$1**
- Production LoRA: **~$10**
- Full fine-tuning: **~$30-35**
- Large-scale: **~$120-125**

---

## Quick Reference Commands

### Essential Commands
```bash
# Check GPU status
nvidia-smi
watch -n 1 nvidia-smi

# Monitor disk usage
df -h

# Monitor system resources
htop

# Check training logs
tail -f /workspace/data/output/*/logs/*.log

# Resume training in tmux
tmux new -s training
tmux attach -t training

# Compress and backup checkpoint
tar -czf checkpoint.tar.gz /workspace/data/output/*/checkpoint-*/
```

### File Transfers
```bash
# Upload to pod
scp -P PORT file.txt root@POD_ID.runpod.io:/workspace/data/

# Download from pod
scp -P PORT root@POD_ID.runpod.io:/workspace/data/model.pt ./

# Sync directory
rsync -avz -e "ssh -p PORT" ./data/ root@POD_ID.runpod.io:/workspace/data/
```

---

## Summary & Best Practices

### ✅ Do's

- ✅ Use network volumes for all important data
- ✅ Save checkpoints every 500-1000 steps
- ✅ Use spot instances for cost savings
- ✅ Monitor GPU utilization and adjust batch sizes
- ✅ Use tmux/screen for persistent sessions
- ✅ Test with small dataset first
- ✅ Stop pods when not in use
- ✅ Keep only necessary checkpoints
- ✅ Use mixed precision (bf16/fp16)
- ✅ Backup final models to cloud storage

### ❌ Don'ts

- ❌ Store data only on pod storage (ephemeral)
- ❌ Leave pods running idle
- ❌ Skip checkpointing on spot instances
- ❌ Use on-demand without trying spot first
- ❌ Over-provision GPUs
- ❌ Run training in foreground without tmux
- ❌ Ignore VRAM usage warnings
- ❌ Keep all checkpoints indefinitely
- ❌ Start with largest GPU without testing

---

## Support & Resources

### RunPod Documentation
- [Official Docs](https://docs.runpod.io/)
- [GPU Pod Guide](https://docs.runpod.io/pods/overview)
- [Network Volumes](https://docs.runpod.io/storage/network-volumes)

### Community
- [RunPod Discord](https://discord.gg/runpod)
- [RunPod Forum](https://community.runpod.io/)

### Invoice-AI Documentation
- `README.md` - Quick start
- `LLAMAFACTORY_GUIDE.md` - Training methods
- `UPGRADE_NOTES.md` - Technical details
- `MIGRATION_CHECKLIST.md` - Deployment guide

---

**Happy Training on RunPod! 🚀**

*For questions or issues, open an issue on GitHub or consult RunPod support.*
