# 🚀 RunPod Serverless Setup Complete Guide

**Tujuan:** Deploy UVIP-AI ke RunPod Serverless dengan RTX 4090 untuk AI inference cepat tanpa cold start delay.

---

## 📋 **Prerequisites Checklist**

- [ ] Akun RunPod di https://runpod.io (daftar gratis)
- [ ] Credit card/crypto untuk payment (~$5 minimum deposit)
- [ ] GitHub repository berisi code UVIP-AI (push semua file sebelumnya)
- [ ] Git installed di komputer lokal

---

## 🔍 **Step 1: Pilih GPU Type**

RunPod menawarkan beberapa pilihan GPU. Untuk UVIP-AI saya pilihkan yang optimal:

| GPU Type | VRAM | Price/Hour | Best For | Recommendation |
|----------|------|------------|----------|----------------|
| **RTX 4090** | 24GB | $0.28/hr | **AI Inference** ✅ | **SELECT THIS!** |
| RTX A6000 | 48GB | $0.65/hr | Heavy training | Too expensive for your use case |
| RTX A5000 | 24GB | $0.45/hr | Medium workloads | Not worth the extra cost |
| A100 80GB | 80GB | $1.50/hr | Enterprise training | Way overkill |
| T4 | 16GB | $0.13/hr | Basic inference | Insufficient VRAM for full pipeline |

**✅ RECOMMENDATION: RTX 4090**

Alasan:
- 24GB VRAM cukup untuk SegFormer-B5 + DINOv2-Large bersamaan
- Ampere architecture sangat cepat untuk inference
- Harga per GB VRAM terbaik
- Supports FP16 (hemat memory, speedup 2×)

---

## 💾 **Step 2: Network Volume Backup**

RunPod punya 2 jenis storage:

1. **Container Disk** (30GB default): Ephemeral, hilang saat pod stop/delete
2. **Network Volume**: Persistent, data awet meski pod mati

**✅ RECOMMENDATION:** Gunakan Network Volume untuk persistence

Cara setup:
1. Dashboard → Network Volumes → Create New Volume
2. Name: `uvip-data`
3. Size: 100 GB (cukup untuk models + photos cache)
4. Click "Create"

Volume ini akan otomatis mount di: `/network-volume/uvip-data`

---

## 🖥️ **Step 3: Pilih Base Image**

Untuk template, pilih salah satu:

| Template | CUDA Version | Python | When to Use |
|----------|--------------|--------|-------------|
| **runpod/pytorch:2.1.0-py3.10-cuda12.1** | 12.1 | 3.10 | ✅ RECOMMENDED (latest stable) |
| runpod/torchvision:2.0.1-py3.9-cuda11.8 | 11.8 | 3.9 | If compatibility issues |
| nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 | 12.1 | System | Manual setup (advanced) |

**✅ RECOMMENDATION: runpod/pytorch:2.1.0-py3.10-cuda12.1**

Alasan:
- PyTorch 2.1 sudah optimized untuk RTX 4090 Ada Lovelace
- CUDA 12.1 support penuh
- Python 3.10 (compatible dengan semua dependencies)
- Pre-installed dengan drivers NVIDIA

---

## 🔑 **Step 4: Generate API Key**

Dari dashboard RunPod:

1. Login ke https://www.runpod.io/console/serverless
2. Click profil Anda (top-right corner)
3. Select **"Settings"**
4. Tab **"API Keys"**
5. Klik **"Generate New Key"**
6. Copy key tersebut (format: `rpa_xxxxxxxxxxxxxxxxxxx...`)
7. **JANGAN SHARED atau commit ke GitHub!**

Store API key aman di password manager.

---

## 🎯 **Step 5: Deploy Serverless Endpoint**

### **A. Buat Endpoint Baru**

Dashboard → Serverless → Endpoints → **Deploy Endpoint**

Fill in form ini:

| Field | Value |
|-------|-------|
| **Endpoint Name** | `uvip-ai-serverless` |
| **Deployment Type** | **Queue-based Serverless** ⭐ |
| **GPU Count** | 1 |
| **GPU Type** | NVIDIA GeForce RTX 4090 |
| **Template** | runpod/pytorch:2.1.0-py3.10-cuda12.1 |
| **Dockerfile Path** | /Dockerfile |
| **Minimum Instances** | 0 (scale-to-zero untuk hemat) |
| **Maximum Instances** | 10 (auto-scale saat traffic tinggi) |
| **Instance Type** | Serverless (default) |

### **B. Advanced Settings**

Expand "Advanced settings":

**Environment Variables:**

```bash
RUNPOD_SERVERLESS=true
RUNPOD_RUNTIME=true
UVIP_LOW_VRAM_MODE=true
UVIP_USE_FP16=true
UVIP_DEVICE=auto
HF_HOME=/app/models/hf_cache
DATA_DIR=/app/data
MODELS_DIR=/app/models
NETWORK_VOLUME_PATH=/network-volume/uvip-data
```

**Port Configuration:**
- Container Port: 8001 (HTTP)
- Protocol: HTTP

**Network Volume Mounts:**
- Volume: uvip-data
- Mount Path: /network-volume/uvip-data

**Storage:**
- Container Storage: 50 GB (models cache)

Click **"Create"** button

---

## 🔐 **Step 6: Configure Firewall & Security**

Default security already secure, tapi tambahkan layer perlindungan:

1. **Enable SSL/TLS** (HTTPS):
   - Dashboard → Endpoint Settings → HTTPS
   - Enable auto-cert via Let's Encrypt
   
2. **Rate Limiting** (optional):
   ```python
   # Add di src/uvip_ai/api/main.py
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address())
   ```

3. **Authentication** (optional):
   ```python
   # Require API key for production
   @app.post("/serverless/invoke")
   async def invoke(payload: dict, x_api_key: Optional[str] = Header(None)):
       if x_api_key != YOUR_API_KEY:
           raise HTTPException(401, "Unauthorized")
       return ...
   ```

---

## 📡 **Step 7: Test Deployment**

### **Method 1: Direct cURL Test**

Di komputer lokal (bukan di Pod!), buka terminal:

```bash
curl https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: rpa_YOUR_API_KEY_HERE" \
  -d '{
    "image": "",
    "latitude": -7.976,
    "longitude": 112.630
  }'
```

Ganti `YOUR_ENDPOINT_ID` dengan ID endpoint Anda dari dashboard.
Ganti API key dengan key yang Anda generate.

### **Method 2: Python Test Script**

Save sebagai `test_runpod.py`:

```python
import requests
import json

CONFIG = {
    "ENDPOINT_ID": "YOUR_ENDPOINT_ID",
    "API_KEY": "rpa_YOUR_API_KEY"
}

url = f"https://api.runpod.ai/v2/{CONFIG['ENDPOINT_ID']}/runsync"

headers = {
    "Content-Type": "application/json",
    "Authorization": CONFIG["API_KEY"]
}

payload = {
    "image": "",  # Empty triggers validation error but proves connectivity
    "latitude": -7.976,
    "longitude": 112.630
}

print("="*70)
print("Testing RunPod Serverless Endpoint")
print(f"URL: {url}")
print("="*70)

response = requests.post(url, headers=headers, json=payload, timeout=120)

print(f"\nStatus Code: {response.status_code}")
print("\nResponse:")
print(json.dumps(response.json(), indent=2))
print("="*70)
```

Run: `python test_runpod.py`

### **Expected Response:**

Jika serverless working correctly:

```json
{
  "success": false,
  "error": "Image must be hex-encoded string or binary"
}
```

Ini OKAY! Berarti:
- ✅ Serverless handler terhubung
- ✅ Pipeline berjalan
- ⚠️ Perlu valid image data untuk hasil real

Untuk tes dengan image nyata, convert image ke hex:

```python
with open("path/to/test.jpg", "rb") as f:
    image_bytes = f.read()
    image_hex = image_bytes.hex()

payload["image"] = image_hex
```

---

## 🔍 **Step 8: Monitor & Logs**

### **Check Worker Status**

Dashboard → Your endpoint → Tab "Workers"

You should see:
- Running workers: 0 (idle mode, scale-to-zero working!)
- Active workers recommended: Auto-suggest based on load

### **View Request Logs**

Dashboard → Your endpoint → Tab "Requests"

See all incoming/outgoing requests with:
- Timestamp
- Processing time
- Status (success/failed)
- Input/output data

### **Error Debugging**

If encountering errors:

1. Check "Metrics" tab for patterns
2. View detailed logs in "Logs" tab
3. Check "Bulk Requests" for custom test payloads

---

## 💰 **Cost Management**

### **Pricing Breakdown (RTX 4090 @ $0.28/hr)**

| Usage Scenario | Hours/Month | Cost |
|----------------|-------------|------|
| Idle (always available) | 720 hrs | $201.60 |
| 10% utilization (172 hrs) | 172 hrs | $48.16 |
| 5% utilization (86 hrs) | 86 hrs | $24.08 |
| Spot usage (10 hrs/month) | 10 hrs | $2.80 |

### **Optimization Strategies:**

1. **Set Minimum Instances = 0** (scale-to-zero)
   - Saves money when no requests
   - First request will have ~5-10 sec cold start

2. **Use Spinning Policy**:
   - Set to "Spinning Up to Min" in advanced settings
   - Prevents constant scaling up/down

3. **Schedule Auto-Scale**:
   ```python
   # Cron-style schedule to scale up during peak hours
   0 8 * * 1-5 runpodctl scale uvip-ai --min 1 --max 5  # Work days 8am
   0 18 * * 1-5 runpodctl scale uvip-ai --min 0 --max 10  # Work days 6pm
   ```

4. **Terminate Unused Resources:**
   - Delete old network volumes monthly
   - Monitor and clean up large model caches

---

## 🔄 **Step 9: Update & Redeploy**

To update code after changes:

### **Method A: Via Dashboard UI**

1. Dashboard → Your endpoint → "Manage" dropdown
2. Select **"Rebuild"**
3. Choose branch from repo (usually `main`)
4. Wait for rollout completion (check progress bar)

### **Method B: Via Git Push**

If configured webhook integration:

```bash
git add .
git commit -m "Update serverless handler"
git push origin main

# RunPod auto-detects and rebuilds automatically!
```

### **Method C: Force Rebuild**

If rollback needed:

1. Dashboard → Endpoint → "Releases"
2. Select previous version
3. Click "Deploy This Release"

---

## 🆘 **Troubleshooting Common Issues**

### **Issue 1: Cold Start Too Slow (>30 seconds)**

**Symptoms:** First request takes forever, subsequent fast

**Causes:**
- Models not pre-loaded in container
- Large initial payload

**Fixes:**
```python
# At module level (not function), preload critical models
from src.uvip_ai.features.dinov2 import MODEL_CACHE
_ = MODEL_CACHE.emb_model  # Trigger lazy loading
_ = MODEL_CACHE.seg_model
```

Or increase minimum instances to 1-2 for always-warmed state.

### **Issue 2: OOM (Out of Memory) Errors**

**Symptoms:** Worker crashes with "CUDA out of memory"

**Causes:**
- VRAM insufficient for batch processing
- Multiple concurrent heavy requests

**Fixes:**
```python
# Reduce batch size in extract_features
batch_size = 1  # instead of 4 or 8

# Use lower precision
model_type = torch.float16  # FP16 instead of float32

# Or enable low VRAM mode
UVIP_LOW_VRAM_MODE=true
```

### **Issue 3: Connection Timeout**

**Symptoms:** Request hangs indefinitely

**Causes:**
- Network latency
- Worker not responsive

**Fixes:**
```python
# Increase timeout in client code
timeout = 300  # 5 minutes instead of 30

# Retry logic
for attempt in range(3):
    try:
        response = requests.post(url, timeout=timeout)
        break
    except Timeout:
        continue
```

### **Issue 4: Permission Denied on Files**

**Symptoms:** Can't read/write files

**Causes:**
- Volume permissions wrong
- User context mismatch

**Fixes:**
```bash
# Fix volume permissions
chmod 755 /network-volume/uvip-data
chown -R root:root /network-volume/uvip-data
```

---

## ✅ **Final Deployment Checklist**

Before going to production:

- [ ] Endpoint created with RTX 4090 selected
- [ ] Network volume mounted (100GB+)
- [ ] Environment variables set correctly
- [ ] Dockerfile path: `/Dockerfile` correct
- [ ] Queue-based deployment selected
- [ ] API key generated and stored securely
- [ ] Health check passes (`https://api.runpod.ai/v2/{id}/runsync`)
- [ ] Test with sample image works
- [ ] Logs being captured
- [ ] Monitoring alerts configured (optional)
- [ ] Budget sufficient ($50+ deposit recommended)
- [ ] Rollout complete (100% workers updated)

---

## 🚀 **Production Deployment Commands**

After everything working:

### **1. Get Production URL**

```
Your endpoint will be at:
https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync
```

Example: `https://api.runpod.ai/v2/abc123xyz789/runsync`

### **2. Integration Example**

```python
import requests
from PIL import Image
from io import BytesIO

def process_image_with_runpod(image_path, lat, lng):
    """Process single image through RunPod serverless"""
    
    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = f.read().hex()
    
    # Prepare payload
    payload = {
        "image": image_data,
        "latitude": lat,
        "longitude": lng
    }
    
    # Call RunPod
    response = requests.post(
        "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync",
        headers={
            "Content-Type": "application/json",
            "Authorization": "YOUR_API_KEY"
        },
        json=payload,
        timeout=300
    )
    
    result = response.json()
    
    if result.get("success"):
        return {
            "beauty_score": result["beauty_score"],
            "safety_score": result["safety_score"],
            # ... other fields
        }
    else:
        raise ValueError(f"Inference failed: {result.get('error')}")

# Usage example
result = process_image_with_runpod(
    "/path/to/photo.jpg",
    -7.976,  # latitude
    112.630  # longitude
)

print(result)
```

---

## 📊 **Performance Expectations**

With RTX 4090 Serverless:

| Metric | Value | Notes |
|--------|-------|-------|
| **Cold Start Time** | 8-15 seconds | First request only |
| **Warm Request Time** | 2-4 seconds | All subsequent requests |
| **Image Throughput** | ~15-20 images/min | Single instance |
| **Max Concurrency** | 10 parallel instances | Auto-scales based on demand |
| **Average Latency** | <700ms per segment | Full pipeline |
| **VRAM Usage** | ~18GB dedicated | Leaves headroom for safety |

Compared to CPU-only VPS:
- ⚡ **10-15× faster** inference
- 🎯 **Consistent performance** regardless of load
- 💰 **Pay-per-use** vs fixed monthly cost

---

## 🎯 **Next Steps After Setup**

1. **Configure Mobile App Integration:**
   - Update mobile app to call RunPod endpoint
   - Add error handling for timeouts
   - Implement retry logic

2. **Set Up Monitoring:**
   - Configure Grafana/Prometheus for metrics
   - Alert on high error rates or latency spikes
   - Track cost usage daily

3. **Backup Strategy:**
   - Regular snapshots of network volume
   - Export trained models to S3/GCS
   - Version control for model configs

4. **Load Testing:**
   - Simulate expected traffic patterns
   - Test auto-scaling behavior
   - Identify bottlenecks before launch

5. **Documentation:**
   - Document API endpoints for team
   - Create runbooks for common issues
   - Train operations team on RunPod console

---

## 💬 **Support Resources**

- **RunPod Docs:** https://docs.runpod.io
- **Community Forum:** https://forum.runpod.io
- **Discord:** https://discord.gg/runpod
- **Support Email:** support@runpod.io
- **Status Page:** https://status.runpod.io

---

**Last Updated:** September 2026  
**Version:** 1.0 (Complete guide for UVIP-AI on RunPod Serverless)

