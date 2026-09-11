# ⚡ Quick Serverless Setup (5 Minutes)

## 📝 **Step-by-Step Guide**

### **1️⃣ Push Code to GitHub**

```bash
git add src/uvip_ai/runpod_serverless.py \
      src/uvip_ai/serverless_entry.py \
      RUNPOD_SERVERLESS_GUIDE.md

git commit -m "Add RunPod Serverless support"
git push origin main
```

---

### **2️⃣ Create Network Volume** (One-time only)

```
RunPod Console → Network Volumes → Create New Volume
Name: uvip-data
Size: 100 GB
```

---

### **3️⃣ Deploy Serverless Endpoint**

Go to: https://www.runpod.io/console/serverless/endpoint/create

Fill in these settings:

| Setting | Value |
|---------|-------|
| **Endpoint Name** | `uvip-ai-serverless` |
| **Template** | `runpod-pytorch-latest-cuda12.1` |
| **Dockerfile Path** | `/Dockerfile` |
| **Network Volume** | Select `uvip-data` from dropdown |
| **GPU** | NVIDIA RTX 4090 × 1 |
| **Instance Type** | **Serverless (Queue)** ⭐ |
| **Minimum Instances** | 0 |
| **Maximum Instances** | 10 |

**Advanced Settings → Environment Variables:**

```
RUNPOD_SERVERLESS=true
UVIP_LOW_VRAM_MODE=true
UVIP_USE_FP16=true
HF_HOME=/app/models/hf_cache
```

Click **"Create"** button

---

### **4️⃣ Test Deployment**

Wait ~2 minutes for provisioning, then test:

```bash
# Get your endpoint URL from dashboard:
https://YOUR_ENDPOINT_ID.rp.runpod.net

# Test health check:
curl https://YOUR_ENDPOINT_ID.rp.runpod.net/health

# Expected response:
{
  "status": "ok",
  "timestamp": "2024-09-10Txx:xx:xx.xxxxxx"
}
```

✅ **SUCCESS!** You're now running on RunPod Serverless!

---

## 💰 **Cost Savings Example**

**Before (Standard Pod):** $204.50/month (always-on)  
**After (Serverless):** $30-50/month (pay-per-use)  
**Savings:** ~85% cheaper!

---

## 🎯 **What Makes This Work**

My code includes these key components:

1. ✅ **`runpod_serverless.py`** - Main handler with model caching
2. ✅ **`serverless_entry.py`** - Entry point that RunPod requires
3. ✅ **Auto-detection** - Works in both Standard & Serverless modes
4. ✅ **Model Singleton** - Loads once, reuses forever
5. ✅ **Low VRAM mode** - Optimized for RTX 4090

No additional changes needed after pushing!

---

## 🔄 **Update Existing Standard Pod**

If you already have a Standard Pod running:

1. Create new Serverless endpoint following steps above
2. Both can run simultaneously (no conflict!)
3. Migrate traffic gradually
4. Delete Standard Pod when Serverless is stable

---

**Questions?** Check `RUNPOD_SERVERLESS_GUIDE.md` for detailed troubleshooting.
