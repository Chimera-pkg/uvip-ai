# 🚀 UVIP-AI Model Deployment Guide (Free Tier)

## 📊 Model Performance Summary

```
✅ Beauty:  R² = 0.9749 (97.5% accuracy)
✅ Safety:  R² = 0.9749 (97.5% accuracy)
✅ Comfort: R² = 0.9749 (97.5% accuracy)
✅ UVI:     R² = 0.9965 (99.7% accuracy)
```

Models siap di-deploy!

---

## 🆓 Free Tier Options Comparison

| Platform | Free Tier | GPU | RAM | Storage | Duration | Best For |
|----------|-----------|-----|-----|---------|----------|----------|
| **Hugging Face Spaces** | ✅ Yes | T4 (15GB) | 16GB | 50GB | Unlimited | Production-ready |
| **Google Colab** | ✅ Yes | T4 (15GB) | 12GB | 100GB | 12h/session | Testing |
| **Kaggle Notebooks** | ✅ Yes | P100/T4 | 13GB | 20GB | 30h/week | Testing |
| **Alibaba Cloud** | ⚠️ Limited | No GPU | 1GB | 40GB | 3 months | CPU-only |
| **Oracle Cloud** | ✅ Yes | No GPU | 24GB | 200GB | Always free | CPU-only |
| **Render.com** | ✅ Yes | No GPU | 512MB | - | 75h/month | API testing |
| **Railway** | ⚠️ $5 credit | No GPU | 1GB | - | One-time | Quick test |

---

## 🎯 Recommended: Hugging Face Spaces (Best Free Option)

### Why HF Spaces?
- ✅ **Free GPU** (T4 15GB) - perfect for SegFormer + DINOv2
- ✅ **Unlimited runtime** - no session limits
- ✅ **Auto-scaling** - handles traffic spikes
- ✅ **Free SSL** - HTTPS included
- ✅ **Public API** - easy integration
- ✅ **50GB storage** - enough for models

### Step-by-Step Deployment

#### 1. Create Hugging Face Account
```bash
# Sign up at https://huggingface.com/join
# Verify email
```

#### 2. Create New Space
```bash
# Go to: https://huggingface.com/new-space
# Settings:
#   - Space name: uvip-ai-model
#   - SDK: Docker
#   - License: MIT
#   - Hardware: T4 small (free)
#   - Visibility: Public
```

#### 3. Prepare Files

Create these files locally:

**`app.py`** (rename from model_service.py)
```python
# Copy content from scripts/model_service.py
```

**`requirements.txt`**
```txt
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
torch==2.1.0
torchvision==0.16.0
transformers==4.36.2
numpy==1.26.2
Pillow==10.1.0
xgboost==2.0.3
scikit-learn==1.3.2
```

**`Dockerfile`**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 7860

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

**`README.md`**
```markdown
---
title: UVIP-AI Model Service
emoji: 🏙️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# UVIP-AI Model Service

Urban Visual Perception AI - Model Inference Service

## API Endpoints

### POST /predict
Upload image and get perception scores.

**Request:** multipart/form-data with `file` field
**Response:** JSON with segmentation features + predictions

### POST /predict/batch
Upload multiple images.

### GET /health
Health check endpoint.
```

#### 4. Upload to Hugging Face

```bash
# Install Git LFS
git lfs install

# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/uvip-ai-model
cd uvip-ai-model

# Copy files
cp /path/to/model_service.py app.py
cp /path/to/requirements.txt .
cp /path/to/Dockerfile .
cp /path/to/README.md .

# Copy models
mkdir -p models/perception
cp /path/to/models/perception/*.pkl models/perception/

# Add and commit
git add .
git commit -m "Initial deployment"
git push
```

#### 5. Wait for Build
```bash
# Check build status at:
# https://huggingface.co/spaces/YOUR_USERNAME/uvip-ai-model

# Build time: ~5-10 minutes
# First run will download models (~2GB)
```

#### 6. Test API
```bash
# Your API will be available at:
# https://YOUR_USERNAME-uvip-ai-model.hf.space

# Test health endpoint
curl https://YOUR_USERNAME-uvip-ai-model.hf.space/health

# Test prediction
curl -X POST https://YOUR_USERNAME-uvip-ai-model.hf.space/predict \
  -F "file=@test_image.jpg"
```

---

## 🥈 Alternative: Google Colab (Quick Testing)

### Pros
- ✅ Free GPU (T4)
- ✅ Easy setup
- ✅ Interactive testing

### Cons
- ❌ 12-hour session limit
- ❌ Not suitable for production
- ❌ Public URL changes each session

### Setup Steps

#### 1. Open Colab Notebook
```bash
# Go to: https://colab.research.google.com/
# Create new notebook
```

#### 2. Install Dependencies
```python
!pip install fastapi uvicorn python-multipart torch torchvision transformers numpy Pillow xgboost scikit-learn pyngrok
```

#### 3. Upload Models
```python
from google.colab import files

# Upload model files
uploaded = files.upload()

# Create directory
!mkdir -p models/perception

# Move files
!mv *.pkl models/perception/
```

#### 4. Create and Run Service
```python
# Copy model_service.py code here
# Add ngrok tunnel

from pyngrok import ngrok

# Start API in background
!nohup uvicorn model_service:app --host 0.0.0.0 --port 8000 &

# Create tunnel
public_url = ngrok.connect(8000)
print(f"Public URL: {public_url}")
```

#### 5. Test API
```python
# Use the ngrok URL
# Example: https://abc123.ngrok.io/predict
```

---

## 🥉 Alternative: Alibaba Cloud (3-Month Free Trial)

### Free Tier Details
- ✅ **ECS Instance**: 1 vCPU, 1GB RAM
- ✅ **Storage**: 40GB SSD
- ✅ **Duration**: 3 months
- ❌ **No GPU** - CPU only (slow inference)

### Setup Steps

#### 1. Sign Up
```bash
# Go to: https://www.alibabacloud.com/free
# Sign up with email
# Verify account
```

#### 2. Create ECS Instance
```bash
# Console → Elastic Compute Service → Create Instance
# Settings:
#   - Region: Singapore (closest to Indonesia)
#   - Instance Type: ecs.t6-c1m1.large (free tier)
#   - Image: Ubuntu 22.04
#   - Storage: 40GB SSD
#   - Network: Assign public IP
```

#### 3. SSH to Server
```bash
ssh root@YOUR_SERVER_IP
```

#### 4. Install Dependencies
```bash
# Update system
apt update && apt upgrade -y

# Install Python
apt install -y python3-pip python3-venv git

# Create app directory
mkdir -p /opt/uvip-ai
cd /opt/uvip-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
```

#### 5. Upload and Install
```bash
# Upload files via SCP
scp -r scripts/ root@YOUR_SERVER_IP:/opt/uvip-ai/
scp -r models/ root@YOUR_SERVER_IP:/opt/uvip-ai/

# Install requirements
pip install -r scripts/requirements.txt
```

#### 6. Run Service
```bash
# Run with nohup
nohup uvicorn scripts.model_service:app --host 0.0.0.0 --port 8000 &

# Or use systemd (recommended)
```

#### 7. Configure Firewall
```bash
# Open port 8000
ufw allow 8000/tcp
ufw reload
```

#### 8. Test API
```bash
curl http://YOUR_SERVER_IP:8000/health
```

**Note**: CPU-only inference will be slow (~10-30 seconds per image vs 1-2 seconds with GPU)

---

## 🔗 Integration with Backend

### Update Backend to Call Model Service

Add this to your backend (`http://103.92.214.110:8001`):

**`app/services/ai_service.py`**
```python
import httpx
from fastapi import UploadFile

MODEL_SERVICE_URL = "https://YOUR_USERNAME-uvip-ai-model.hf.space"

async def analyze_photo(photo_id: str, photo_file: UploadFile):
    """Call AI model service and save results."""
    
    # Call model service
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{MODEL_SERVICE_URL}/predict",
            files={"file": (photo_file.filename, await photo_file.read(), photo_file.content_type)}
        )
    
    if response.status_code != 200:
        raise Exception(f"Model service error: {response.text}")
    
    result = response.json()
    
    # Save segmentation results
    segmentation_data = {
        "photo_id": photo_id,
        "model_name": "segformer-b0",
        "vegetation_pct": result["segmentation"]["vegetation_pct"],
        "building_pct": result["segmentation"]["building_pct"],
        "road_pct": result["segmentation"]["road_pct"],
        "sidewalk_pct": result["segmentation"]["sidewalk_pct"],
        "sky_pct": result["segmentation"]["sky_pct"],
        "signage_pct": result["segmentation"]["signage_pct"],
        "vehicle_pct": result["segmentation"]["vehicle_pct"],
        "pedestrian_pct": result["segmentation"]["pedestrian_pct"],
        "street_furniture_pct": result["segmentation"]["street_furniture_pct"],
        "green_coverage_pct": result["segmentation"]["green_coverage_pct"],
        "building_coverage_pct": result["segmentation"]["building_coverage_pct"],
        "sky_visibility_pct": result["segmentation"]["sky_visibility_pct"],
        "walkability_ratio": result["segmentation"]["walkability_ratio"],
        "visual_clutter_index": result["segmentation"]["visual_clutter_index"],
        "inference_time_ms": result["inference_time_ms"]
    }
    
    # Save to database (using your existing CRUD)
    segmentation = await create_segmentation(segmentation_data)
    
    # Save perception predictions
    prediction_data = {
        "photo_id": photo_id,
        "segmentation_id": segmentation.id,
        "model_version": "v1.0",
        "beauty_score": result["predictions"]["beauty_score"],
        "safety_score": result["predictions"]["safety_score"],
        "comfort_score": result["predictions"]["comfort_score"],
        "uvi_score": result["predictions"]["uvi_score"],
        "inference_time_ms": result["inference_time_ms"]
    }
    
    prediction = await create_prediction(prediction_data)
    
    return {
        "segmentation": segmentation,
        "prediction": prediction
    }
```

**Add endpoint to backend:**
```python
@app.post("/photos/{photo_id}/analyze")
async def analyze_photo_endpoint(
    photo_id: str,
    file: UploadFile = File(...)
):
    """Analyze photo with AI model."""
    result = await analyze_photo(photo_id, file)
    return result
```

---

## 📊 Testing Checklist

### 1. Test Model Service
```bash
# Health check
curl https://YOUR_MODEL_SERVICE_URL/health

# Single prediction
curl -X POST https://YOUR_MODEL_SERVICE_URL/predict \
  -F "file=@test_image.jpg"

# Batch prediction
curl -X POST https://YOUR_MODEL_SERVICE_URL/predict/batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

### 2. Test Backend Integration
```bash
# Upload photo
curl -X POST http://103.92.214.110:8001/photos/ \
  -F "file=@test_image.jpg" \
  -F "mission_id=YOUR_MISSION_ID"

# Analyze photo
curl -X POST http://103.92.214.110:8001/photos/PHOTO_ID/analyze \
  -F "file=@test_image.jpg"

# Get results
curl http://103.92.214.110:8001/photos/PHOTO_ID
```

### 3. Verify Data Flow
```bash
# Check segmentation results
curl http://103.92.214.110:8001/segmentation-results/?photo_id=PHOTO_ID

# Check predictions
curl http://103.92.214.110:8001/perception-predictions/?photo_id=PHOTO_ID
```

---

## 🎯 Recommendation

### For Testing (Now)
**Use Google Colab** - Quick setup, free GPU, perfect for testing integration

### For Production (Later)
**Use Hugging Face Spaces** - Free GPU, unlimited runtime, production-ready

### If You Need Full Control
**Use Alibaba Cloud** - 3-month free trial, but CPU-only (slow)

---

## 📝 Next Steps

1. ✅ **Deploy to Hugging Face Spaces** (15 minutes)
2. ✅ **Test API endpoints** (5 minutes)
3. ✅ **Integrate with backend** (30 minutes)
4. ✅ **Test end-to-end flow** (10 minutes)
5. ✅ **Monitor performance** (ongoing)

---

## 🆘 Troubleshooting

### Model Service Issues
```bash
# Check logs (HF Spaces)
# Click "Logs" tab in your Space

# Common errors:
# - Out of memory: Reduce batch size
# - Model not found: Check file paths
# - CORS error: Update allowed origins
```

### Backend Integration Issues
```bash
# Check backend logs
# Test model service directly first
# Verify network connectivity
```

### Slow Inference
```bash
# CPU-only: Expected (10-30s per image)
# GPU: Should be fast (1-2s per image)
# Check batch size
# Optimize model loading
```

---

## 💰 Cost Summary

| Option | Cost | Duration | GPU | Recommendation |
|--------|------|----------|-----|----------------|
| Hugging Face Spaces | **FREE** | Unlimited | T4 | ✅ **Best choice** |
| Google Colab | **FREE** | 12h/session | T4 | ✅ Good for testing |
| Alibaba Cloud | **FREE** | 3 months | ❌ No | ⚠️ CPU only |
| Oracle Cloud | **FREE** | Always | ❌ No | ⚠️ CPU only |
| Railway | $5 credit | One-time | ❌ No | ⚠️ Limited |

**Winner: Hugging Face Spaces** 🏆
