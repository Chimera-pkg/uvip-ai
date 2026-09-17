# UVIP-AI Server Deployment Guide

## 📋 Table of Contents
1. [Persiapan Model](#1-persiapan-model)
2. [Pilihan Server](#2-pilihan-server)
3. [Setup Railway (Recommended)](#3-setup-railway)
4. [Setup DigitalOcean](#4-setup-digitalocean)
5. [API Integration](#5-api-integration)
6. [Testing](#6-testing)
7. [Monitoring](#7-monitoring)

---

## 1. Persiapan Model

### Export Model dari Kaggle

```python
# Di Kaggle notebook, setelah training selesai
import pickle
import json
from pathlib import Path

# Create export directory
export_dir = Path('model_export')
export_dir.mkdir(exist_ok=True)

# Save models
for name in ['beauty', 'safety', 'comfort', 'uvi']:
    model_path = f'uvip-ai/models/perception/{name}_xgb.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Export ke format yang bisa di-load
    export_path = export_dir / f'{name}_model.pkl'
    with open(export_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Exported: {export_path}")

# Save metrics
with open('uvip-ai/models/perception/metrics.json', 'r') as f:
    metrics = json.load(f)

with open(export_dir / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Download sebagai zip
import zipfile
with zipfile.ZipFile('model_export.zip', 'w') as zipf:
    for file in export_dir.glob('*'):
        zipf.write(file, arcname=file.name)

print("✓ Download model_export.zip dari Kaggle Output")
```

### Struktur Project untuk Deploy

```
uvip-ai-server/
├── app.py                    # FastAPI application
├── requirements.txt          # Dependencies
├── models/
│   ├── beauty_model.pkl
│   ├── safety_model.pkl
│   ├── comfort_model.pkl
│   └── uvi_model.pkl
├── feature_extractor.py      # SegFormer + DINOv2
├── Dockerfile               # Container config
├── railway.json             # Railway config
└── tests/
    └── test_api.py          # API tests
```

---

## 2. Pilihan Server

### Option A: Railway (Recommended - $5/bulan)

**Pros:**
- Deploy dalam 5 menit
- Auto-scaling
- Free SSL
- Easy monitoring
- GitHub integration

**Cons:**
- $5/bulan (tapi worth it)
- Limited customization

### Option B: DigitalOcean ($6/bulan)

**Pros:**
- Full control
- Better performance
- More customization
- SSH access

**Cons:**
- Setup lebih complex
- Manual deployment
- Need sysadmin knowledge

### Option C: Render (Free Tier)

**Pros:**
- Free untuk testing
- Easy deploy
- Auto-scaling

**Cons:**
- Sleep setelah 15 menit idle
- Limited resources
- Cold start lambat

---

## 3. Setup Railway

### Step 1: Install Railway CLI

```bash
# Windows
npm install -g @railway/cli

# Login
railway login
```

### Step 2: Create Project Structure

```bash
mkdir uvip-ai-server
cd uvip-ai-server

# Copy models dari Kaggle
# (download model_export.zip dari Kaggle, extract ke folder models/)
```

### Step 3: Create FastAPI App

**app.py:**
```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
from PIL import Image
import io
from typing import Dict, List
import time

from feature_extractor import extract_features

app = FastAPI(
    title="UVIP-AI API",
    description="Urban Visual Identity Perception AI",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
print("Loading models...")
models = {}
for name in ['beauty', 'safety', 'comfort', 'uvi']:
    with open(f'models/{name}_model.pkl', 'rb') as f:
        models[name] = pickle.load(f)
    print(f"✓ Loaded {name} model")

class PredictionResponse(BaseModel):
    filename: str
    predictions: Dict[str, float]
    processing_time: float

@app.get("/")
async def root():
    return {
        "service": "UVIP-AI API",
        "version": "1.0.0",
        "status": "healthy",
        "models": list(models.keys())
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict beauty, safety, comfort, dan UVI scores dari gambar.
    """
    start_time = time.time()
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Extract features
        features = extract_features(image)
        
        # Predict
        predictions = {}
        for name, model in models.items():
            pred = model.predict(features.reshape(1, -1))[0]
            predictions[name] = round(float(pred), 2)
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            filename=file.filename,
            predictions=predictions,
            processing_time=round(processing_time, 3)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict multiple images sekaligus.
    """
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            features = extract_features(image)
            
            predictions = {}
            for name, model in models.items():
                pred = model.predict(features.reshape(1, -1))[0]
                predictions[name] = round(float(pred), 2)
            
            results.append({
                "filename": file.filename,
                "predictions": predictions
            })
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results, "total": len(results)}
```

**feature_extractor.py:**
```python
import torch
import numpy as np
from PIL import Image
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    AutoImageProcessor,
    Dinov2Model
)

# Load models (cached)
print("Loading feature extraction models...")
seg_processor = SegformerImageProcessor.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
seg_model = SegformerForSemanticSegmentation.from_pretrained(
    'nvidia/segformer-b0-finetuned-ade-512-512'
).to('cuda' if torch.cuda.is_available() else 'cpu').to(torch.float16).eval()

dinov2_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
dinov2_model = Dinov2Model.from_pretrained(
    'facebook/dinov2-base'
).to('cuda' if torch.cuda.is_available() else 'cpu').to(torch.float16).eval()

print("✓ Feature extraction models loaded")

def extract_features(image: Image.Image) -> np.ndarray:
    """
    Extract 773 features dari image (5 seg metrics + 768 DINOv2 embeddings).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # SegFormer features
    inputs = seg_processor(images=image, return_tensors="pt").to(device)
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
    
    with torch.no_grad():
        outputs = seg_model(**inputs)
    
    seg_map = outputs.logits.argmax(dim=1)[0].cpu().numpy()
    
    unique_classes = len(np.unique(seg_map))
    main_class_ratio = np.max(np.bincount(seg_map.flatten())) / seg_map.size
    
    seg_features = np.array([
        unique_classes,
        main_class_ratio,
        unique_classes / 150,
        1 - main_class_ratio,
        unique_classes / 50
    ])
    
    # DINOv2 features
    inputs = dinov2_processor(images=image, return_tensors="pt").to(device)
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
    
    with torch.no_grad():
        outputs = dinov2_model(**inputs)
    
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    
    # Concatenate
    features = np.concatenate([seg_features, embedding])
    
    return features
```

**requirements.txt:**
```txt
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
torch==2.1.0
torchvision==0.16.0
transformers==4.35.2
numpy==1.26.2
Pillow==10.1.0
xgboost==2.0.2
scikit-learn==1.3.2
```

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**railway.json:**
```json
{
  "build": {
    "builder": "DOCKERFILE"
  },
  "deploy": {
    "numReplicas": 1,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

### Step 4: Deploy ke Railway

```bash
# Initialize Railway project
railway init

# Set environment variables
railway variables set PYTHON_VERSION=3.11

# Deploy
railway up

# Open browser
railway open
```

### Step 5: Test API

```bash
# Test health endpoint
curl https://your-app.up.railway.app/health

# Test prediction
curl -X POST "https://your-app.up.railway.app/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

---

## 4. Setup DigitalOcean

### Step 1: Create Droplet

1. Login ke DigitalOcean
2. Create → Droplets
3. Choose:
   - Image: Ubuntu 22.04 LTS
   - Plan: Basic ($6/month - 1GB RAM, 1 vCPU)
   - Region: Singapore (closest to Indonesia)
   - Authentication: SSH Key

### Step 2: SSH ke Server

```bash
ssh root@your_server_ip
```

### Step 3: Setup Server

```bash
# Update system
apt update && apt upgrade -y

# Install Python
apt install -y python3-pip python3-venv

# Install Docker (optional)
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Create app directory
mkdir -p /opt/uvip-ai
cd /opt/uvip-ai
```

### Step 4: Deploy App

```bash
# Clone atau copy files
# (upload files dari lokal)

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run app
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Step 5: Setup Nginx + SSL

```bash
# Install Nginx
apt install -y nginx

# Create config
nano /etc/nginx/sites-available/uvip-ai
```

**Nginx config:**
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

```bash
# Enable site
ln -s /etc/nginx/sites-available/uvip-ai /etc/nginx/sites-enabled/
nginx -t
systemctl restart nginx

# Install SSL (Let's Encrypt)
apt install -y certbot python3-certbot-nginx
certbot --nginx -d your-domain.com
```

### Step 6: Setup Systemd Service

```bash
nano /etc/systemd/system/uvip-ai.service
```

**Service file:**
```ini
[Unit]
Description=UVIP-AI API
After=network.target

[Service]
User=root
WorkingDirectory=/opt/uvip-ai
ExecStart=/opt/uvip-ai/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable service
systemctl daemon-reload
systemctl enable uvip-ai
systemctl start uvip-ai

# Check status
systemctl status uvip-ai
```

---

## 5. API Integration

### Frontend Integration (JavaScript)

```javascript
// Upload dan predict
async function predictImage(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('https://your-api.com/predict', {
        method: 'POST',
        body: formData
    });
    
    const data = await response.json();
    console.log(data.predictions);
    // {beauty: 7.2, safety: 6.8, comfort: 5.9, uvi: 4.3}
}
```

### Python Integration

```python
import requests

url = "https://your-api.com/predict"
files = {"file": open("test.jpg", "rb")}

response = requests.post(url, files=files)
predictions = response.json()["predictions"]

print(f"Beauty: {predictions['beauty']}")
print(f"Safety: {predictions['safety']}")
print(f"Comfort: {predictions['comfort']}")
print(f"UVI: {predictions['uvi']}")
```

---

## 6. Testing

### Unit Tests

**tests/test_api.py:**
```python
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
    # Create dummy image
    from PIL import Image
    import io
    
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert all(k in data["predictions"] for k in ["beauty", "safety", "comfort", "uvi"])

def test_batch_predict():
    # Create multiple dummy images
    from PIL import Image
    import io
    
    files = []
    for i in range(3):
        img = Image.new('RGB', (224, 224), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        files.append(("files", (f"test_{i}.jpg", img_bytes, "image/jpeg")))
    
    response = client.post("/predict/batch", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
```

### Run Tests

```bash
# Install test dependencies
pip install pytest httpx

# Run tests
pytest tests/ -v
```

### Load Testing

```bash
# Install hey
go install github.com/rakyll/hey@latest

# Load test (100 requests, 10 concurrent)
hey -n 100 -c 10 -m POST -H "Content-Type: multipart/form-data" \
    -F "file=@test.jpg" \
    https://your-api.com/predict
```

---

## 7. Monitoring

### Railway Monitoring

```bash
# View logs
railway logs

# View metrics
railway metrics
```

### DigitalOcean Monitoring

```bash
# View logs
journalctl -u uvip-ai -f

# View system resources
htop

# View Nginx logs
tail -f /var/log/nginx/access.log
```

### Custom Monitoring

```python
# Add to app.py
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

---

## 8. Cost Breakdown

### Railway ($5/bulan)
- 1GB RAM, 1 vCPU
- 100GB bandwidth
- Auto-scaling
- **Total: $5/bulan**

### DigitalOcean ($6/bulan)
- 1GB RAM, 1 vCPU
- 1TB bandwidth
- Full control
- **Total: $6/bulan**

### Render (Free)
- 512MB RAM
- 100GB bandwidth
- Sleep setelah 15 menit
- **Total: $0/bulan** (tapi limited)

---

## 9. Troubleshooting

### Model Loading Error
```bash
# Check model files
ls -lh models/

# Verify model format
python -c "import pickle; pickle.load(open('models/beauty_model.pkl', 'rb'))"
```

### Memory Error
```bash
# Reduce batch size
# Use CPU instead of GPU
# Optimize feature extraction
```

### Slow Response
```bash
# Enable caching
# Use Redis for feature cache
# Optimize model inference
```

---

## 10. Next Steps

1. ✅ Setup server (Railway/DigitalOcean)
2. ✅ Deploy API
3. ✅ Test endpoints
4. ✅ Integrate with frontend
5. ✅ Setup monitoring
6. ✅ Collect user feedback
7. ✅ Retrain model dengan data baru
8. ✅ Scale server jika perlu

---

## Kesimpulan

**Dataset 431 foto:**
- ✅ Cukup untuk MVP/prototype
- ⚠️ Perlu augmentasi (431 → 2155)
- ❌ Tidak cukup untuk production (butuh 2000+)

**Server recommendation:**
- 🏆 **Railway** - $5/bulan, easy deploy
- 💰 **DigitalOcean** - $6/bulan, full control
- 🆓 **Render** - Free tier, limited

**Timeline:**
- Setup server: 1-2 jam
- Deploy API: 30 menit
- Testing: 1 jam
- **Total: 3-4 jam**
