# 📡 RunPod SSH Access Guide

**Important:** RunPod Serverless endpoints **TIDAK menggunakan SSH**! 

Serverless menggunakan model **Queue-based API** yang dipanggil via HTTP requests. Tidak ada shell access ke worker instances.

---

## ❌ Kenapa SSH Tidak Ada di Serverless?

1. **Serverless = Ephemeral**: Worker instances dibuat/dihancurkan otomatis
2. **Security**: No direct access to infrastructure
3. **Scalability**: Multiple workers running simultaneously
4. **Managed Service**: All managed via RunPod dashboard & API

---

## ✅ Cara Akses runpod serverless:

### **Method 1: Dashboard UI** (Recommended)

Dashboard → Serverless → Your Endpoint → Tabs:
- **Overview**: Basic info & status
- **Logs**: View execution logs
- **Requests**: Monitor incoming/outgoing requests  
- **Workers**: Check active worker count
- **Metrics**: Performance metrics
- **Bulk Requests**: Test endpoint via browser

### **Method 2: API Calls** (Production)

```bash
curl https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: YOUR_API_KEY" \
  -d '{"image": "", "latitude": 0, "longitude": 0}'
```

### **Method 3: RunPod CLI** (Advanced)

```bash
# Install CLI
pip install runpodctl

# List your endpoints
runpodctl list endpoints

# Get endpoint details
runpodctl get endpoints --name uvip-ai-serverless

# Delete endpoint
runpodctl delete endpoint --name uvip-ai-serverless
```

---

## 🔄 Jika Butuh Shell Access: Gunakan **Standard Pod**

Jika Anda butuh SSH/shell access, deploy **Standard Pod** bukan Serverless:

1. Dashboard → Pods → Deploy Pod
2. Select "Standard" mode (NOT "Serverless")
3. Template: `runpod/pytorch:2.1.0-py3.10-cuda12.1`
4. GPU: RTX 4090
5. After pod created: Dashboard → Terminal (web terminal)

Atau via SSH jika enabled:
```bash
ssh root@YOUR_POD_IP -p PORT
```

**Trade-off:** Standard Pod always-running ($200+/month) vs Serverless pay-per-use ($30-50/month).

---

## 📊 Quick Comparison

| Feature | Serverless | Standard Pod |
|---------|------------|--------------|
| **SSH Access** | ❌ NO | ✅ YES |
| **Web Terminal** | ⚠️ Limited (logs only) | ✅ Full shell access |
| **Cost Model** | Pay per request | Always-on hourly |
| **Cold Start** | 5-15 seconds first request | 0 seconds (always running) |
| **Best For** | Variable traffic | Constant high load |

---

*For full SSH setup, see RUNPOD_STANDARD_DEPLOY.md (if you deleted it)*
