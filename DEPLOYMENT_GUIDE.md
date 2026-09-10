# 🚀 UVIP-AI Deployment Guide ke RunPod RTX 4090

## ⚠️ ERROR DIASHBOARD RUNPOD: "Could not find runpod.serverless.start()"

Ini terjadi karena project menggunakan **Docker Container mode**, bukan **Serverless mode**.

---

## ✅ SOLUSI TERBAIK: Standard Pod + Docker Compose

### **Prerequisites:**
1. Akun RunPod: https://runpod.io
2. Install CLI: `pip install runpodctl`
3. API Key dari dashboard → Settings → API Keys

---

## 📝 Step 1: Buat Pod via CLI

```bash
# Login ke RunPod
runpodctl config --apiKey YOUR_API_KEY_HERE

# Create pod RTX 4090
runpodctl create pod \
  --name uvip-production \
  --gpuType "NVIDIA GeForce RTX 4090" \
  --gpuCount 1 \
  --volumeSize 200 \
  --imageName "runpod/pytorch:2.1.0-py3.10-cuda12.1"
```

### **Output yang diharapkan:**
```
Created pod with name: uvip-production
Status: Creating...
Endpoint: http://xxx.xxx.xxx.xxx:xxxx
```

Tunggu ~5 menit hingga status berubah menjadi **"Running"**

---

## 🔗 Step 2: Connect ke Pod

### **Opsi A: Web Terminal (Paling Mudah)**

1. Buka dashboard: https://www.runpod.io/console/serverless
2. Klik pada pod `uvip-production`
3. Klik tombol **"Terminal"** di bagian atas

### **Opsi B: SSH via Command Line**

```bash
# Dapatkan IP address
runpodctl get endpoints --name uvip-production

# SSH ke pod (port biasanya 8443 atau sesuai dashboard)
ssh root@YOUR_POD_IP -p YOUR_PORT
```

---

## 🛠️ Step 3: Deploy Code & Build

Di dalam terminal/pod SSH:

```bash
# 1. Clone repository
git clone <your-repo-url>
cd uvip-ai

# 2. Copy environment configuration
cp .env.example .env

# 3. Build Docker image
docker build -t uvip-ai:latest .

# 4. Run with GPU support
docker run --gpus all \
  -p 8001:8001 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  --name uvip-api \
  uvip-ai:latest
```

**Atau gunakan docker-compose.yml (lebih baik):**

```bash
cp .env.example .env
docker-compose up -d --build
```

---

## ✅ Step 4: Verify Deployment

```bash
# Check container logs
docker logs uvip-api

# Test health endpoint
curl http://localhost:8001/health

# Expected response:
# {"status": "ok", "timestamp": "2024-xx-xxTxx:xx:xx"}
```

---

## 💰 Cost Management

### **Hentikan Pod saat Tidak Digunakan:**

```bash
# Stop pod (berhenti billing!)
runpodctl stop --name uvip-production

# Atau via dashboard: Serverless → Pods → uvip-production → Stop
```

### **Start Kembali:**

```bash
# Start pod
runpodctl start --name uvip-production

# Via dashboard: klik "Start" button
```

---

## 🚨 Troubleshooting

### **Problem: Container tidak start**
```bash
# Lihat error log
docker logs uvip-api

# Restart container
docker restart uvip-api
```

### **Problem: GPU tidak terdeteksi**
```bash
# Verify GPU inside container
docker exec uvip-api nvidia-smi

# Ensure --gpus all flag used when running container
```

### **Problem: Model tidak load**
```bash
# Clear HuggingFace cache
rm -rf ./models/hf_cache

# Model akan download otomatis saat pertama run
```

---

## 🎯 Quick Commands Cheat Sheet

```bash
# Pod management
runpodctl list pods              # Show all pods
runpodctl stop --name xxx        # Stop pod
runpodctl start --name xxx       # Start pod  
runpodctl delete --name xxx      # Delete pod permanently

# Container commands
docker ps                        # List running containers
docker logs uvip-api             # View logs
docker exec -it uvip-api bash    # Enter container shell

# Cost saving
runpodctl stop --name uvip-production   # Stop billing
runpodctl start --name uvip-production  # Resume
```

---

## 📊 Estimated Costs

| Configuration | Price/Month |
|---------------|-------------|
| RTX 4090 @ $0.40/hr (always-on) | ~$288/mo |
| RTX 4090 @ $0.40/hr (10 hrs/day) | ~$120/mo |
| RTX 4090 @ $0.40/hr (weekends only) | ~$57/mo |

**Tip:** Always stop pod via dashboard/CLI saat development selesai untuk hemat biaya!

---

*Document last updated: September 2026*
