# 🚀 UVIP AI - Deployment Guide for Vast.ai GTX 1070

## 📋 Server Specifications (Your Server)
- **GPU:** NVIDIA GTX 1070 (8GB VRAM, CUDA 12.8) ✅
- **CPU:** Xeon E5-2620 v3 @ 2.4GHz (12 threads)
- **RAM:** Sufficient for inference
- **Storage:** 16GB NVMe SSD
- **Cost:** $0.051/hour ⚡ **SUPER CHEAP!**

---

## 🎯 Quick Start (5 Steps)

### Step 1: Setup Project on Server
```bash
# SSH into your server
ssh root@182.224.239.168

# Navigate to workspace
cd /workspace

# Run setup script
chmod +x scripts/setup_vast_ai.sh
./scripts/setup_vast_ai.sh
```

### Step 2: Upload Trained Models
```bash
# Upload models from local to server
python scripts/upload_models.py \
    --local-dir models/perception \
    --remote-path /app/models/perception \
    --key-file ~/.ssh/vastai_private_key
```

### Step 3: Deploy API Server
```bash
# Start the API server
chmod +x scripts/deploy_api.sh
./scripts/deploy_api.sh

# Server will run and you can CTRL+C to stop monitoring
# Server continues running in background
```

### Step 4: Test API
```bash
# Health check
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs

# Test prediction (from another terminal)
curl -X POST "http://localhost:8000/predict" \
    -F "file=@data/extracted/photos/KAYUTANGAN/KAYUTANGAN_ST-01.jpg"
```

### Step 5: Monitor Logs
```bash
# Access logs
tail -f logs/access.log

# Error logs  
tail -f logs/error.log

# Server output
cat api_server.log
```

---

## 🔧 Complete Setup Commands

### Alternative: Manual Setup (if auto-script fails)

```bash
# 1. Verify environment
python --version
python -c "import torch; print(torch.__version__)"

# 2. Install dependencies
pip install -r requirements.txt
pip install uvicorn gunicorn

# 3. Create directories
mkdir -p models/perception data/raw data/extracted test_outputs logs

# 4. Upload models
# (Use SCP or upload_models.py)

# 5. Start server
gunicorn src.api.main:app \
    --bind 0.0.0.0:8000 \
    --workers 2 \
    --timeout 120 &
```

---

## 📦 Files Structure

```
/app/
├── scripts/
│   ├── setup_vast_ai.sh       # Initial setup
│   ├── deploy_api.sh          # Deploy API
│   └── upload_models.py       # Upload helper
├── models/
│   └── perception/
│       ├── beauty_model.pkl
│       ├── safety_model.pkl
│       ├── comfort_model.pkl
│       └── uvi_model.pkl
├── src/
│   └── api/
│       └── main.py           # FastAPI server
├── logs/
│   ├── access.log            # Request logs
│   └── error.log             # Error logs
└── test_outputs/             # Results storage
```

---

## 🔐 Security Notes

1. **Never commit SSH keys** to git (already in .gitignore)
2. Use strong password for root (or disable root login)
3. Configure firewall if needed
4. Add API rate limiting in production

---

## 📊 Performance Expectations

With GTX 1070 (8GB VRAM):
- **Single image processing:** ~15-25 seconds
- **Batch size:** 4-8 images at once
- **Memory usage:** ~4-6 GB GPU RAM
- **Throughput:** ~10-20 predictions/minute

---

## 💰 Cost Calculation

Current rate: **$0.051/hour**

| Runtime | Cost |
|---------|------|
| 1 hour | $0.05 |
| 1 day | $1.22 |
| 1 week | $8.57 |
| 1 month | $36.72 |

**Recommendation:** Set up automatic billing and shut down when not in use.

---

## 🆘 Troubleshooting

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size or process single images
```bash
# Edit predict_batch.py or adjust in config
```

### Issue: "Model not found"
**Solution:** Check models are uploaded correctly
```bash
ls -la /app/models/perception/
```

### Issue: "Port 8000 already in use"
**Solution:** Change port
```bash
gunicorn ... --bind 0.0.0.0:8001
```

### Issue: "Import error"
**Solution:** Re-run setup script
```bash
./scripts/setup_vast_ai.sh
```

---

## 🎓 Fine-tuning on Server

If you want to fine-tune SegFormer-B5 with Indonesian data:

```bash
# Prepare training data (upload to /app/data/fine_tuning/)
# Then run:
python scripts/fine_tune_segformer.py \
    --train-dir /app/data/fine_tuning/images \
    --mask-dir /app/data/fine_tuning/masks \
    --output-dir /app/models/finetuned_indonesia \
    --batch-size 8 \
    --num-epochs 15 \
    --learning-rate 1e-5
```

---

## 🌐 External Access

To access from outside:
1. Update security group rules on Vast.ai
2. Allow traffic on port 8000
3. Access at: `http://182.224.239.168:8000`

---

## 📞 Support

For issues:
1. Check logs first: `logs/error.log`
2. Review health status: `http://localhost:8000/health`
3. Consult this guide or contact support

---

## 🎉 Success Checklist

- [x] Server connected via SSH
- [x] Virtual environment activated
- [x] All dependencies installed
- [x] Trained models uploaded
- [x] API server running on port 8000
- [x] Health check passed
- [x] API docs accessible
- [x] Test prediction successful
- [x] Logs being captured
- [x] Auto-restart configured

**Congratulations! Your UVIP AI is deployed! 🚀**
