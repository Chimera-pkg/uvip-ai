# ⚡ Direct Run Guide - UVIP-AI without Docker

## 🎯 Problem Solved: Docker Daemon Not Available in Pod

Karena pod RunPod Anda tidak support Docker-in-Docker, kita akan jalankan aplikasi langsung dengan Python.

---

## 📝 Step-by-Step Deployment (Tanpa Docker)

### **STEP 1: Install Dependencies**

```bash
cd /home/uvip-ai

# Install PyTorch dengan CUDA support (udah pre-installed di template torch)
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### **STEP 2: Setup Environment**

```bash
# Copy environment configuration
cp .env.example .env

# Verify .env created
ls -la .env
```

### **STEP 3: Start API Server Directly**

```bash
# Option A: Single worker mode (simple)
PYTHONPATH=./src uvicorn src.uvip_ai.api.main:app \
  --host 0.0.0.0 \
  --port 8001 \
  --workers 1

# Option B: Multi-worker mode (recommended for production)
PYTHONPATH=./src uvicorn src.uvip_ai.api.main:app \
  --host 0.0.0.0 \
  --port 8001 \
  --workers 4 \
  --log-level info
```

**Untuk background running:**
```bash
# Run as background service
nohup PYTHONPATH=./src uvicorn src.uvip_ai.api.main:app \
  --host 0.0.0.0 \
  --port 8001 \
  --workers 4 > api.log 2>&1 &

# Get PID
echo $!

# Check if running
ps aux | grep uvicorn
```

### **STEP 4: Verify Deployment**

```bash
# Test health endpoint
curl http://localhost:8001/health

# Expected response:
{
  "status": "ok", 
  "timestamp": "2024-09-10Txx:xx:xx.xxxxxx"
}

# View logs
tail -f api.log
```

### **STEP 5: Auto-start on Restart**

Jika perlu auto-restart saat pod restart, buat systemd service:

```bash
# Create systemd service file
cat > /etc/systemd/system/uvip-api.service << 'EOF'
[Unit]
Description=UVIP-AI FastAPI Service
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/uvip-ai
Environment="PYTHONPATH=/home/uvip-ai"
ExecStart=/usr/bin/python3 -m uvicorn src.uvip_ai.api.main:app --host 0.0.0.0 --port 8001 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable uvip-api
sudo systemctl start uvip-api

# Check status
sudo systemctl status uvip-api

# View logs
sudo journalctl -u uvip-api -f
```

---

## 🎮 Quick Commands Reference

### **Start/Stop Management**

```bash
# Start directly
PYTHONPATH=./src uvicorn src.uvip_ai.api.main:app --host 0.0.0.0 --port 8001 &

# Stop process
pkill -f uvicorn

# Or kill by port
lsof -ti:8001 | xargs kill -9
```

### **Monitor Logs**

```bash
# Real-time logs
tail -f api.log

# Search errors
grep -i error api.log

# Last 50 lines with context
tail -n 50 api.log | grep -A5 -B5 ERROR
```

### **Check Process Status**

```bash
# Is uvicorn running?
ps aux | grep uvicorn

# Is port 8001 open?
netstat -tuln | grep 8001

# GPU utilization
nvidia-smi

# Memory usage
free -h
```

---

## 🔧 Troubleshooting

### **Problem: Module not found**
```bash
# Fix PYTHONPATH
export PYTHONPATH=/home/uvip-ai/src:$PYTHONPATH

# Re-run
python -c "import sys; print(sys.path)"
```

### **Problem: Port already in use**
```bash
# Find and kill process
lsof -ti:8001 | xargs kill -9

# Or use different port
uvicorn src.uvip_ai.api.main:app --port 8002
```

### **Problem: GPU not detected**
```bash
# Verify CUDA visible
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'Devices: {torch.cuda.device_count()}')"

# Should output: CUDA available: True, Devices: 1
```

### **Problem: Models fail to download**
```bash
# Set cache location
mkdir -p models/hf_cache
export HF_HOME=models/hf_cache

# Download specific model
python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov2-small')"
```

---

## 💾 Data Persistence Strategy

### **Mount Network Volume Manually**

Network volume di `/network-volume/UVIP-AI_volume` sudah mounted. Mount ke aplikasi:

```bash
# Create symbolic links
ln -sf /network-volume/UVIP-AI_volume/models models
ln -sf /network-volume/UVIP-AI_volume/data data

# Verify
ls -la models
ls -la data
```

Or mount at runtime:

```bash
export DATA_DIR=/network-volume/UVIP-AI_volume/data
export MODELS_DIR=/network-volume/UVIP-AI_volume/models
```

---

## 🚀 Alternative: Use Screen/Tmux Session

Untuk avoid disconnection saat close terminal:

```bash
# Install tmux (if not available)
apt-get update && apt-get install -y tmux

# Create new session
tmux new-session -s uvip-api

# Run uvicorn
PYTHONPATH=./src uvicorn src.uvip_ai.api.main:app --host 0.0.0.0 --port 8001 --workers 4

# Detach session: Ctrl+B, then D
# Reattach later: tmux attach-session -s uvip-api
```

---

## ✅ Deployment Checklist

Before deploy complete:
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file created from `.env.example`
- [ ] PYTHONPATH set correctly
- [ ] Uvicorn running on port 8001
- [ ] Health check passes (`curl http://localhost:8001/health`)
- [ ] CUDA/GPU accessible
- [ ] Network volume mounted
- [ ] Auto-restart configured (systemd or tmux)

---

**Need help?** Check `README.md` or ask for assistance!
