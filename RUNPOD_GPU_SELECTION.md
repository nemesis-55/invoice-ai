# RunPod GPU Quick Reference

Quick decision guide for selecting the right GPU configuration on RunPod for training MiniCPM-V-4.5.

---

## 🎯 Decision Tree

### Question 1: What's your budget?
- **<$5 total**: Budget Option (Single GPU)
- **$5-15**: Recommended Option (2x GPUs)
- **$30-50**: Premium Option (Multi-GPU)
- **$100+**: Enterprise Option (Large-scale)

### Question 2: What's your training method?
- **LoRA**: Can work with Budget or Recommended
- **Full Fine-tuning**: Need Premium or Enterprise

### Question 3: How urgent is it?
- **Days are fine**: Use Budget (slower but cheaper)
- **Hours preferred**: Use Recommended (balanced)
- **ASAP**: Use Premium (fast)

---

## 📊 GPU Configuration Matrix

| Config Name | GPUs | VRAM | Spot $/hr | Training | Cost for 10K steps |
|------------|------|------|-----------|----------|-------------------|
| **Budget** | 1x RTX 4090 | 24GB | $0.40 | LoRA | ~$1.20 (3hrs) |
| **Recommended** | 2x A40 | 96GB | $0.80 | LoRA | ~$2.00 (2.5hrs) |
| **Premium** | 2x A100 80GB | 160GB | $2.50 | Full | ~$15 (6hrs) |
| **Enterprise** | 4x A100 80GB | 320GB | $5.00 | Full | ~$20 (4hrs) |

---

## 🎓 Use Case Recommendations

### Scenario 1: "I'm experimenting / learning"
- **Recommended**: Budget (1x RTX 4090)
- **Why**: Cheap to iterate, good for testing
- **Cost**: ~$1-2 per experiment
- **Training**: LoRA only

### Scenario 2: "I need production-ready model"
- **Recommended**: Recommended (2x A40)
- **Why**: Faster training, better generalization
- **Cost**: ~$6-10 for full training
- **Training**: LoRA (recommended) or start of Full

### Scenario 3: "I need maximum accuracy"
- **Recommended**: Premium (2x A100 80GB)
- **Why**: Full fine-tuning capability
- **Cost**: ~$30-40 for full training
- **Training**: Full fine-tuning

### Scenario 4: "I'm training continuously / research"
- **Recommended**: Enterprise (4x A100 80GB)
- **Why**: Fastest training, can handle large batches
- **Cost**: $100-150+ for extensive training
- **Training**: Full fine-tuning with large batches

---

## 💡 Pro Tips

### Save Money
1. **Always use Spot instances** (50-80% savings)
2. **Start small**: Test with Budget config first
3. **Checkpoint frequently**: Every 500 steps
4. **Stop immediately**: When not actively training
5. **Clean old checkpoints**: Keep only last 3

### Optimize Performance
1. **Monitor GPU usage**: Use `nvidia-smi`
2. **Adjust batch size**: Fill 80-90% of VRAM
3. **Use mixed precision**: bf16 or fp16
4. **Gradient accumulation**: Increase if OOM
5. **Network volume**: Store data on fast storage

### Avoid Common Mistakes
1. ❌ Leaving pod running idle
2. ❌ Not saving checkpoints on spot instances
3. ❌ Using on-demand without trying spot first
4. ❌ Keeping all checkpoints (wastes money)
5. ❌ Not monitoring actual GPU usage

---

## 🚀 Quick Start Commands

### Deploy Budget Config (1x RTX 4090)
```bash
# On RunPod dashboard:
# 1. Select "Community Cloud"
# 2. Filter: RTX 4090
# 3. Sort by: Lowest price
# 4. Deploy with 100GB network volume
```

### Deploy Recommended Config (2x A40)
```bash
# On RunPod dashboard:
# 1. Select "Community Cloud"
# 2. Filter: A40
# 3. Minimum GPUs: 2
# 4. Deploy with 200GB network volume
```

### Deploy Premium Config (2x A100 80GB)
```bash
# On RunPod dashboard:
# 1. Select "Community Cloud" or "Secure Cloud"
# 2. Filter: A100 80GB
# 3. Minimum GPUs: 2
# 4. Deploy with 200GB network volume
```

---

## 📈 ROI Calculator

### Example: Production LoRA Training

**Scenario**: Fine-tune for invoice extraction (10K steps)

**Option A: Budget (1x RTX 4090)**
- Training time: 3 hours
- Cost: $0.40/hr × 3 = **$1.20**
- Quality: Good
- **Best for**: Testing, learning

**Option B: Recommended (2x A40)**
- Training time: 1.5 hours
- Cost: $0.80/hr × 1.5 = **$1.20**
- Quality: Better (larger batch size)
- **Best for**: Production

**Option C: Premium (2x A100 80GB)**
- Training time: 1 hour
- Cost: $2.50/hr × 1 = **$2.50**
- Quality: Best (full fine-tuning possible)
- **Best for**: Critical applications

**Winner**: Option B (Recommended) - Same cost as Budget but faster and better quality!

---

## ⚠️ Important Notes

### Spot Instance Interruptions
- **Frequency**: Rare but possible
- **Solution**: Auto-resume from checkpoint
- **Prevention**: Checkpoint every 500 steps
- **Risk**: Acceptable for training (not for inference)

### GPU Availability
- **Peak hours**: Harder to find GPUs
- **Solution**: Be flexible with GPU type
- **Alternative**: Use "Secure Cloud" (higher cost)
- **Tip**: Set up pod early in the day

### Network Volume Costs
- **Pricing**: ~$0.10/GB/month
- **Example**: 200GB = $20/month
- **Recommendation**: Delete after training
- **Alternative**: Backup to cloud storage (Azure/S3/GCS)

---

## 📞 When to Use Each Config

### Budget (1x RTX 4090)
✅ Learning and experimentation  
✅ Small datasets (<10K samples)  
✅ LoRA fine-tuning only  
✅ Budget <$5  
❌ Production deployments  
❌ Full fine-tuning  
❌ Time-critical projects  

### Recommended (2x A40)
✅ Production LoRA training  
✅ Medium datasets (10K-50K)  
✅ Balance of cost and performance  
✅ Budget $5-15  
⚠️ Limited full fine-tuning  
❌ Very large batch sizes  

### Premium (2x A100 80GB)
✅ Full fine-tuning  
✅ Large datasets (50K+)  
✅ Production deployments  
✅ Time-critical projects  
✅ Budget $30-50  
⚠️ Overkill for LoRA  

### Enterprise (4x A100 80GB)
✅ Continuous training  
✅ Research projects  
✅ Very large datasets  
✅ Multi-task training  
✅ Budget $100+  
❌ One-off experiments  
❌ Small datasets  

---

## 🎯 Final Recommendation

**For most users starting out:**

1. **Test phase**: Budget (1x RTX 4090) - $1-2
2. **If results good**: Scale to Recommended (2x A40) - $6-10
3. **If need best quality**: Upgrade to Premium (2x A100) - $30-40

**Total investment to production**: $40-50

This gives you validation → production path with minimal waste!

---

See [RUNPOD_TRAINING_GUIDE.md](RUNPOD_TRAINING_GUIDE.md) for complete details.
