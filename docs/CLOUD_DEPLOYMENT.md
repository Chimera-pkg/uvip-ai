# Cloud Deployment Guide

Panduan deploy UVIP-AI ke cloud GPU (RunPod, GCP, AWS) untuk training dan inference.

## Table of Contents
1. [RunPod (Recommended)](#runpod-recommended)
2. [Google Cloud Platform](#google-cloud-platform)
3. [Amazon Web Services](#amazon-web-services)
4. [Training Workflow](#training-workflow)
5. [Cost Optimization](#cost-optimization)

---

## RunPod (Recommended)

RunPod adalah opsi paling cost-effective untuk GPU training.

### Setup

1. **Create Account**
   - Sign up di https://www.runpod.io/
   - Add payment method (credit card/crypto)

2. **Create Pod**
   ```bash
   # Via CLI
   pip install runpodctl
   runpodctl config --apiKey YOUR_API_KEY
   ```

3. **Launch GPU Pod**
   ```bash
   # Recommended: RTX 4090 (24GB VRAM)
   runpodctl create pod \
     --name uvip-training \
     --gpuType "NVIDIA GeForce RTX 4090" \
     --gpuCount 1 \
     --volumeSize 100 \
     --imageName runpod/pytorch:2.1.0-py3.10-cuda12.1 \
     --networkVolumeId YOUR_VOLUME_ID
   ```

4. **Connect to Pod**
   ```bash
   # SSH access
   ssh root@POD_IP -p PORT
   
   # Or use RunPod web terminal
   ```

5. **Deploy Code**
   ```bash
   # Clone repo
   git clone https://github.com/YOUR_ORG/uvip-ai.git
   cd uvip-ai
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Setup environment
   cp .env.example .env
   # Edit .env with your settings
   ```

6. **Run Training**
   ```bash
   # Build dataset (extract features from all photos)
   python scripts/extract_features.py \
     --input data/extracted/photos \
     --output data/training/features.csv
   
   # Train XGBoost models
   python scripts/train_xgboost.py \
     --input data/training/features.csv \
     --labels data/training/labels.csv \
     --output models/perception/
   ```

7. **Download Models**
   ```bash
   # Copy trained models back to local
   scp -P PORT root@POD_IP:/path/to/uvip-ai/models/perception/*.pkl ./models/perception/
   ```

### Cost
- **RTX 4090**: ~$0.40/hour
- **A100 80GB**: ~$1.50/hour
- **H100**: ~$3.00/hour

Untuk training 434 photos:
- Feature extraction: ~2-3 hours = $0.80-$1.20
- XGBoost training: ~30 minutes = $0.20
- **Total**: ~$1.00-$1.50 per training run

---

## Google Cloud Platform

### Setup

1. **Create Project**
   ```bash
   gcloud init
   gcloud projects create uvip-ai-project
   ```

2. **Enable APIs**
   ```bash
   gcloud services enable compute.googleapis.com
   gcloud services enable storage.googleapis.com
   ```

3. **Create VM with GPU**
   ```bash
   # n1-standard-8 + 1x T4 GPU
   gcloud compute instances create uvip-training \
     --machine-type n1-standard-8 \
     --accelerator type=nvidia-tesla-t4,count=1 \
     --image-family pytorch-latest-gpu \
     --image-project deeplearning-platform-release \
     --boot-disk-size 100GB \
     --zone us-central1-a
   ```

4. **Install GPU Drivers**
   ```bash
   # SSH into VM
   gcloud compute ssh uvip-training
   
   # Install drivers (if not pre-installed)
   sudo /opt/deeplearning/install-driver.sh
   ```

5. **Deploy & Train**
   ```bash
   # Same as RunPod steps 5-7
   ```

### Cost
- **n1-standard-8 + T4**: ~$0.60/hour
- **Storage**: $0.02/GB/month
- **Total training**: ~$2.00-$3.00

---

## Amazon Web Services

### Setup

1. **Launch EC2 Instance**
   ```bash
   # g4dn.xlarge (1x T4 GPU)
   aws ec2 run-instances \
     --image-id ami-0123456789abcdef0 \
     --instance-type g4dn.xlarge \
     --key-name my-key \
     --block-device-mappings file://ebs.json
   ```

2. **Connect**
   ```bash
   ssh -i my-key.pem ec2-user@INSTANCE_IP
   ```

3. **Deploy & Train**
   ```bash
   # Same as RunPod steps 5-7
   ```

### Cost
- **g4dn.xlarge**: ~$0.53/hour (on-demand)
- **Spot instance**: ~$0.16/hour (70% cheaper)
- **Total training**: ~$1.50-$2.50

---

## Training Workflow

### Step 1: Prepare Data Locally

```bash
# Extract photos from PDF
python scripts/extract_photos_from_pdf.py

# Fix coordinates
python scripts/fix_manifest_coords.py

# Upload to cloud
rsync -avz data/extracted/ user@cloud:/path/to/uvip-ai/data/extracted/
```

### Step 2: Extract Features (Cloud)

```bash
# On cloud GPU
python scripts/extract_features.py \
  --input data/extracted/photos \
  --output data/training/features.csv \
  --batch-size 8
```

This runs:
- Privacy Guard (YOLOv8n) → blur faces/plates
- SegFormer-B5 → 5 urban metrics
- DINOv2-Large → 1024-d embedding

Output: `data/training/features.csv` with columns:
- `filename, area, point_id, lat, long`
- `seg_green_coverage_pct, ...` (5 metrics)
- `emb_0, emb_1, ..., emb_1023` (1024-d)

### Step 3: Add Labels

```bash
# Merge with survey labels
python scripts/merge_labels.py \
  --features data/training/features.csv \
  --labels data/training/labels.csv \
  --output data/training/dataset.csv
```

### Step 4: Train Models

```bash
python scripts/train_xgboost.py \
  --input data/training/dataset.csv \
  --output models/perception/ \
  --n-folds 5
```

Output:
- `models/perception/beauty_xgb.pkl`
- `models/perception/safety_xgb.pkl`
- `models/perception/comfort_xgb.pkl`
- `models/perception/uvi_xgb.pkl`
- `models/perception/metrics.json`

### Step 5: Download & Deploy

```bash
# Download models to local
scp user@cloud:/path/to/models/perception/*.pkl ./models/perception/

# Test locally
python scripts/predict.py \
  --image data/extracted/photos/KAYUTANGAN/KAYUTANGAN_ST-01.jpg \
  --models models/perception/
```

---

## Cost Optimization

### Use Spot/Preemptible Instances
- **RunPod**: Already spot pricing
- **GCP**: Add `--preemptible` flag (60-91% cheaper)
- **AWS**: Use spot instances (70% cheaper)

### Optimize Training
```bash
# Reduce batch size if OOM
python scripts/extract_features.py --batch-size 4

# Use smaller models for testing
export DINOV2_MODEL=facebook/dinov2-base  # 768-d instead of 1024-d
export SEGFORMER_MODEL=nvidia/segformer-b3-finetuned-cityscapes-1024-1024
```

### Cache Models
```bash
# Download models once, reuse across runs
export HF_HOME=/path/to/cache/huggingface
```

### Use Network Volumes (RunPod)
```bash
# Store models on network volume (persists across pods)
runpodctl create volume --name uvip-models --size 50
runpodctl create pod --networkVolumeId VOLUME_ID ...
```

---

## Monitoring

### GPU Usage
```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Or use nvtop
sudo apt install nvtop
nvtop
```

### Training Progress
```bash
# Monitor training logs
tail -f logs/training.log

# Check metrics
cat models/perception/metrics.json | jq
```

### Cost Tracking
- **RunPod**: Dashboard → Billing
- **GCP**: Billing → Cost Management
- **AWS**: Cost Explorer

---

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
python scripts/extract_features.py --batch-size 2

# Use fp16 mode
export UVIP_USE_FP16=true

# Clear GPU cache between models
python scripts/extract_features.py --clear-cache
```

### Slow Training
```bash
# Use multiple GPUs
python scripts/train_xgboost.py --n-jobs 4

# Optimize XGBoost parameters
python scripts/train_xgboost.py --n-estimators 200 --max-depth 8
```

### Model Not Converging
```bash
# Check data quality
python scripts/validate_dataset.py --input data/training/dataset.csv

# Increase training data (collect more surveys)
# Or use data augmentation
python scripts/augment_data.py --input data/training/dataset.csv --factor 2
```

---

## Next Steps

1. ✅ Code complete (all modules implemented)
2. ✅ Local setup verified (RTX 3060 6GB)
3. ⏳ Prepare survey labels (data/training/labels.csv)
4. ⏳ Rent cloud GPU (RunPod recommended)
5. ⏳ Run training on cloud
6. ⏳ Download models & test locally
7. ⏳ Deploy to production (Docker + API)

---

## Support

- RunPod Discord: https://discord.gg/runpod
- GCP Support: https://cloud.google.com/support
- AWS Support: https://aws.amazon.com/premiumsupport/
