# 🚀 Deploy UVIP-AI ke RunPod RTX 4090 - Step-by-Step

## 📍 Langkah 1: Buka Web Terminal

1. Dashboard → Pods → UVIP-AI
2. Klik tab **"Web terminal"**
3. Enable jika belum aktif

---

## 📍 Langkah 2: Clone Repository & Setup

Copy-paste commands berikut **SATU PER SATU** di terminal:

```bash
# 1. Clone repository (GANTI URL dengan repo Anda sendiri!)
git clone https://github.com/YOUR_USERNAME/uvip-ai.git
cd uvip-ai
```

```bash
# 2. Copy environment configuration
cp .env.example .env
```

```bash
# 3. Install dependencies (jika tidak pakai Docker)
pip install -r requirements.txt
```

---

## 📍 Langkah 3: Build Docker Image

```bash
# Build Docker image dari Dockerfile yang ada
docker build -t uvip-ai:latest .
```

**Output:**
```
✓ Step 1/10: FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
✓ Step 2/10: RUN apt-get update && ...
...
✓ Successfully built uvip-ai:latest
```

---

## 📍 Langkah 4: Run Container with GPU

```bash
# Run container dengan akses GPU full
docker run --gpus all \
  -p 8001:8001 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  --name uvip-api \
  --restart unless-stopped \
  uvip-ai:latest
```

**Parameter penjelas:**
- `--gpus all` → Full GPU access (RTX 4090)
- `-p 8001:8001` → Map port 8001 (akan muncul di HTTP services!)
- `-v models:/app/models` → Persist model cache
- `-v data:/app/data` → Persist dataset

---

## 📍 Langkah 5: Verify Deployment

### **Check container running:**
```bash
docker ps | grep uvip-api
```

**Expected output:**
```
CONTAINER ID   IMAGE          STATUS
abc123def456   uvip-ai:latest Up 1 minute
```

---

### **Test health endpoint:**
```bash
curl http://localhost:8001/health
```

**Expected response:**
```json
{
  "status": "ok",
  "timestamp": "2024-09-10Txx:xx:xx.xxxxxx"
}
```

---

### **View logs (jika ada error):**
```bash
docker logs -f uvip-api
```

---

## 📍 Langkah 6: Test AI Inference

Jika sudah punya test image:

```bash
# Upload test image terlebih dahulu
#scp test.jpg root@YOUR_POD_IP:/path/to/uvip-ai/data/extracted/photos/

# Process dengan API
curl -X POST http://localhost:8001/ai/process \
  -F "file=@/path/to/test/image.jpg" \
  -F "latitude=-7.976" \
  -F "longitude=112.630"
```

---

## 📍 Langkah 7: Update HTTP Services Port

Setelah container running, **port 8001 akan otomatis muncul**:

1. Refresh dashboard RunPod
2. Check **HTTP services** section
3. Port 8001 seharusnya sudah tersedia!

Atau manually add di dashboard:
- Service Name: `uvip-api`
- Port: `8001`
- Protocol: `HTTP`

---

## 💡 Tips & Troubleshooting

### **If container exits immediately:**
```bash
# View full logs
docker logs --tail 100 uvip-api

# Rebuild if Dockerfile changed
docker build -t uvip-ai:latest --no-cache .
```

### **If GPU not detected:**
```bash
# Inside container
docker exec -it uvip-api nvidia-smi

# Should show RTX 4090 info
```

### **If models fail to load:**
```bash
# Clear cache and rebuild
rm -rf ./models/hf_cache
docker-compose down
docker-compose up -d --build
```

### **To stop billing:**
```bash
# Stop container only (cost still active)
docker stop uvip-api

# OR stop entire pod via dashboard
# Serverless → Pods → UVIP-AI → Stop button
```

---

## 📊 Quick Commands Cheat Sheet

```bash
# Monitor
docker ps                     # List containers
docker stats uvip-api         # Resource usage
nvidia-smi                    # GPU utilization

# Logs
docker logs -f uvip-api       # Follow logs
docker logs uvip-api          # Static logs

# Management
docker restart uvip-api       # Restart service
docker pause uvip-api         # Pause container
docker unpause uvip-api       # Unpause

# Cost saving
docker-compose down           # Stop all services
runpodctl stop --name uvip-production  # Stop pod
```

---

## ✅ Final Checklist

Sebelum deploy complete:
- [ ] Git clone successful
- [ ] `.env` file created
- [ ] Docker image built (no errors)
- [ ] Container running (`docker ps`)
- [ ] Health check passing (`/health` endpoint)
- [ ] Port 8001 exposed in HTTP services
- [ ] GPU accessible (`nvidia-smi` inside container)

---

**Need help?** Check `README.md` or `docs/DEPLOYMENT_GUIDE.md` for detailed documentation.
