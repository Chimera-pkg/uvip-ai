# 🚀 RunPod Standard Pod Deployment - RTX 4090 (No Docker Required!)

## 🎯 Why Standard Pod > Serverless?

| Feature | Standard Pod | Serverless |
|---------|--------------|------------|
| **Performance** | ⚡ Fast, zero cold start | ❌ Cold start 10-30s |
| **GPU Access** | ✅ Full RTX 4090 always available | ⚠️ Models reload each time |
| **Complexity** | ✅ Simple script deployment | ❌ Complex handler setup |
| **Stability** | ✅ Always running | ⚠️ Auto-scale delays |
| **Cost** | $0.28/hr (~$200/month) | $0 when idle, but slow |

**Conclusion:** Standard pod lebih cepat, stabil, dan tidak perlu downgrademodel!

---

## 📋 Pre-requisites

1. ✅ Akun RunPod di https://runpod.io
2. ✅ API credit minimal $5
3. ✅ GitHub repository berisi UVIP-AI code

---

## 🎮 STEP-BY-STEP DEPLOYMENT

### **Step 1: Create Standard Pod via Dashboard**

1. **Login ke RunPod Console:**
   ```
   https://www.runpod.io/console/serverless/pods
   ```

2. **Click "Deploy" → "Create Pod"**

3. **Fill Pod Configuration:**

| Setting | Value |
|---------|-------|
| **Pod Name** | `uvip-production` |
| **Template** | `runpod/pytorch:2.1.0-py3.10-cuda12.1` |
| **Compute Type** | Standard (NOT Serverless/Queue!) |
| **GPU** | NVIDIA GeForce RTX 4090 × 1 |
| **Storage** | 200 GB NVMe SSD |
| **Network Volume** | Create new volume `uvip-data` (50GB) |
| **Container Disk** | Auto (use default) |

4. **Advanced Settings:**

✅ **Enable SSH** (untuk akses terminal)  
✅ **Enable Web Terminal** (untuk deploy tanpa SSH)  
✅ **Container Security**: Default (no special settings needed)  

5. **Click "Deploy" button**

Tunggu ~3-5 menit hingga status berubah menjadi **"Running"**

---

### **Step 2: Get Pod Access Info**

Setelah pod ready, dapatkan informasi dari dashboard:

1. **Endpoint URL:** (tidak dipakai untuk Standard Pod)
2. **Web Terminal Link:** (paling mudah!)
3. **SSH IP & Port:** (jika mau via CLI)

Contoh:
```
Pod Status: Running
Endpoint: http://xxx.xxx.xxx.xxx:xxxx
SSH Host: xxx.xxx.xxx.xxx
SSH Port: xxxx
GPU: RTX 4090 ✓
Memory: 62 GB RAM ✓
```

---

### **Step 3: Connect to Pod**

#### **Option A: Web Terminal (RECOMMENDED - Paling Mudah)**

1. Klik pada pod `uvip-production` di dashboard
2. Klik tab **"Terminal"** di bagian atas
3. Browser akan membuka web terminal dalam tab baru
4. Anda sudah di inside pod dengan user `root`

#### **Option B: SSH (Advanced)**

Jika pakai SSH:

```bash
ssh root@YOUR_SSH_HOST -p YOUR_SSH_PORT
# Contoh: ssh root@123.45.67.89 -p 22619
```

Password biasanya ada di dashboard pod details (atau auto-login untuk templates tertentu).

---

### **Step 4: Deploy Code via Script (One Command!)**

Di web terminal atau SSH shell, jalankan command ini:

```bash
# Copy paste SELURUH isi quick-std-pod-deploy.sh ke terminal
# Script akan otomatis melakukan semua step!
```

Atau jalankan per step manual:

```bash
# === STEP 1: Clone Repository ===
cd /home/uvip-ai || mkdir -p /home/uvip-ai && cd /home/uvip-ai

git clone https://github.com/YOUR_USERNAME/uvip-ai.git .
# Ganti URL dengan repo Anda sendiri!

cp .env.example .env

# === STEP 2: Install Dependencies ===
pip install --upgrade pip
pip install -r requirements.txt --quiet

# === STEP 3: Mount Network Volume ===
mkdir -p /network-volume/UVIP-AI_volume/{models,data}
ln -sf /network-volume/UVIP-AI_volume/models ./models
ln -sf /network-volume/UVIP-AI_volume/data ./data
mkdir -p ./logs

# === STEP 4: Verify GPU ===
python3 -c "import torch; print(f'✓ CUDA: {torch.cuda.is_available()}')"
nvidia-smi

# === STEP 5: Start API Server ===
cat > start_api.sh << 'EOF'
#!/bin/bash
cd /home/uvip-ai
export PYTHONPATH=/home/uvip-ai/src
nohup python3 -m uvicorn src.uvip_ai.api.main:app \
  --host 0.0.0.0 \
  --port 8001 \
  --workers 4 \
  --log-level info > logs/api.log 2>&1 &
echo $! > logs/api.pid
EOF

chmod +x start_api.sh
./start_api.sh

# === STEP 6: Wait & Test ===
sleep 30
curl http://localhost:8001/health
```

---

### **Step 5: Verify Deployment**

After script runs, check these:

```bash
# 1. Check if process running
ps aux | grep uvicorn

# 2. Check health endpoint
curl http://localhost:8001/health

# 3. View logs
tail -f logs/api.log
```

Expected response:
```json
{
  "status": "ok",
  "timestamp": "2024-09-10Txx:xx:xx.xxxxxx"
}
```

✅ **Success!** Server sudah running di RTX 4090!

---

## 🧪 Testing the API

### **Test Health Check:**
```bash
curl http://localhost:8001/health
```

### **Test Image Processing:**

First upload image to pod:

```bash
# From local computer (outside pod):
scp /path/to/test/image.jpg root@YOUR_POD_IP:/home/uvip-ai/data/extracted/photos/

# Or use Python HTTP client from inside pod:
python3 -c "
import requests
with open('/home/uvip-ai/path/to/test.jpg', 'rb') as f:
    resp = requests.post('http://localhost:8001/ai/process', files={'file': ('test.jpg', f)})
    print(resp.json())
"
```

Expected JSON response:
```json
{
  "beauty_score": 7.2,
  "safety_score": 6.8,
  "comfort_score": 7.5,
  "uvi_score": 6.9,
  "green_coverage_pct": 35.2,
  "building_coverage_pct": 28.1,
  "walkability_ratio": 0.42,
  "visual_clutter_index": 0.18,
  "sky_visibility_pct": 22.5,
  "processing_time_ms": 3456
}
```

---

## 💰 Cost Management

### **Standard Pod Pricing (RTX 4090):**
- **Rate:** $0.28/hour
- **Monthly:** ~$201.60/bulan (always running 24/7)
- **Storage:** $0.004/GB/hr = $0.16/day (for 50GB volume)

### **Save Costs:**

When not using server:

```bash
# Option 1: Stop pod completely (via dashboard)
# RunPod Console → Pods → uvip-production → Shutdown

# Option 2: Keep VM running but stop service
sudo systemctl stop uvip-api  # If systemd service created
# or
pkill -f uvicorn  # Quick kill (service will need restart on reboot)

# Option 3: Auto-stop schedule (advanced)
# Add to crontab: @daily 02:00 shutdown -h now
```

**Recommendation:** For production traffic, keep always-on ($200/month is worth it for no cold starts!)

---

## 🔧 Management Commands

### **Start/Stop Server:**

```bash
# Start API server
cd /home/uvip-ai && ./start_api.sh

# Stop server gracefully
pkill -f uvicorn

# Restart server
pkill -f uvicorn && ./start_api.sh
```

### **Monitor:**

```bash
# Process status
ps aux | grep uvicorn

# Resource usage
htop
nvidia-smi  # GPU utilization

# Logs
tail -f logs/api.log
journalctl -u uvip-api -f  # if systemd service
```

### **Logs Management:**

```bash
# Rotate old logs
mv logs/api.log logs/api.log.old
./start_api.sh  # Recreates log file

# Clear all logs
rm logs/api.log*
./start_api.sh
```

---

## 🚨 Troubleshooting

### **Problem: 404 Not Found at /health**

**Cause:** Service belum starting atau port wrong

**Fix:**
```bash
# Check if running
ps aux | grep uvicorn

# Check error logs
tail -50 logs/api.log

# Restart server
pkill -f uvicorn && ./start_api.sh
```

---

### **Problem: No module named "torch"**

**Cause:** Dependencies not installed

**Fix:**
```bash
pip install -r requirements.txt --force-reinstall
```

---

### **Problem: CUDA not detected**

**Cause:** GPU drivers not loaded in container

**Fix:**
1. Check pod template has CUDA support (should have)
2. Reboot pod: Runtime → Pods → uvip-production → Delete
3. Recreate pod with same template

---

### **Problem: Out of Memory errors**

**Cause:** VRAM exhaustion from models

**Fix:** Enable low VRAM mode in `.env`:

```bash
# Edit .env
nano .env

# Set these values:
UVIP_LOW_VRAM_MODE=true
UVIP_USE_FP16=true
```

Then restart server:
```bash
pkill -f uvicorn && ./start_api.sh
```

---

### **Problem: Port 8001 not accessible externally**

**Cause:** Firewall or networking issue

**Fix:**
1. Check pod security settings
2. Verify port exposed in template
3. Use public IP from dashboard to access:
   ```bash
   curl http://YOUR_POD_PUBLIC_IP:8001/health
   ```

---

## 📊 Performance Benchmarks

With RTX 4090 + Standard Pod:

| Metric | Result |
|--------|--------|
| **Cold Start** | 0 seconds (always running) |
| **Image Inference** | 2-4 seconds (full pipeline) |
| **API Latency** | <100ms (per request) |
| **GPU Utilization** | ~60% average |
| **VRAM Usage** | ~18GB dedicated |
| **CPU Usage** | ~4 cores peak |

Compared to Serverless:
- ✅ **2-3× faster** (no model reload each time)
- ✅ **Zero cold start** (always warmed up)
- ✅ **More predictable** latency

---

## ✅ Final Checklist

Before deploying to production:

- [ ] Pod status: "Running"
- [ ] RTX 4090 detected (`nvidia-smi`)
- [ ] API server started (`ps aux | grep uvicorn`)
- [ ] Health endpoint responds (`curl localhost:8001/health`)
- [ ] Image processing works (`curl POST /ai/process`)
- [ ] Logs accessible (`tail -f logs/api.log`)
- [ ] Network volume mounted (data persists after restart)
- [ ] Budget sufficient for long-term running

---

## 🎯 Next Steps

1. **Configure DNS domain** (optional)
   - Point custom domain to pod IP
   - Use SSL certificate (Let's Encrypt)

2. **Set up monitoring**
   - Configure Grafana/Prometheus
   - Alert on high latency/errors

3. **Backup models regularly**
   ```bash
   rsync -av /home/uvip-ai/models/ backup-server:/backups/uvip-models/
   ```

4. **Scale horizontally**
   - Add more pods for load balancing
   - Use Nginx or HAProxy as reverse proxy

---

*Document last updated: September 2026*
