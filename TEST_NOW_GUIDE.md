# 🧪 TEST NOW - Quick Guide for Your UVIP-AI Serverless

## 🎯 Your Current Status (From Dashboard)

```
✅ Endpoint: uvip-ai
✅ Type: Queue-based Serverless
✅ Status: Running  
✅ Workers: 2 active
✅ Rollout: 43% complete
⚡ Cost: $0.00016/second
```

---

## ⚡ **3 MINUTE TEST METHOD** (Browser Only)

### Step 1: Go to "Bulk Requests" Tab

In your dashboard, click **"Bulk Requests"** tab (next to Overview).

You'll see a dropdown with options like:
- `POST /runsync` ⭐ USE THIS ONE
- `POST /run`
- `GET /health`
- etc.

### Step 2: Click Dropdown → Select `POST /runsync`

This will show you a pre-filled cURL command.

### Step 3: Click Purple "Run in Browser" Button

To the right of the cURL command, there's a purple button:

```
[ Run in browser ] ← CLICK THIS!
```

### Step 4: Wait & Get Results

Browser will execute the request and show response in 5-30 seconds.

**Expected Response:**
```json
{
  "success": true,
  "beauty_score": 7.2,
  "safety_score": 6.8,
  ...
}
```

If empty image error: ✅ Still means server is working!

---

## 🐍 **PYTHON TEST METHOD** (Local Computer)

### 1️⃣ Generate API Key

In dashboard "Quick start" section:
```
Click [Generate key]
Copy the key (starts with rp_...)
```

### 2️⃣ Save test.py

```python
import requests
import json

ENDPOINT_ID = "46ifkgzb20xva0"
API_KEY = "PASTE_YOUR_API_KEY_HERE"

url = f"https://{ENDPOINT_ID}.rp.runpod.net/serverless/invoke"
headers = {"Authorization": f"ApiKey {API_KEY}"}

payload = {
    "image": "",  # Empty for now
    "latitude": -7.976,
    "longitude": 112.630
}

print("Sending test...")
response = requests.post(url, headers=headers, json=payload, timeout=120)

print(f"\nStatus: {response.status_code}")
print(json.dumps(response.json(), indent=2))
```

### 3️⃣ Run

```bash
python test.py
```

---

## 🌐 **DIRECT URL TEST**

Open browser at:
```
https://46ifkgzb20xva0.rp.runpod.ai/health
```

Or with curl:
```bash
curl https://46ifkgzb20xva0.rp.runpod.ai/health
```

---

## 📊 What You're Seeing Now

Dashboard shows:
- `2 running workers` ✅ Good!
- `0 jobs in progress` ✅ Idle
- `3 active workers recommended` 🔔 Scale up if needed

Cost is **$0.00016/second** ≈ **$0.58/hour** when running

---

## ✅ Success Indicators

### ✅ If you see:
- Status 200 or 201
- JSON response with scores
- Processing time shown

**→ SERVERLESS WORKING PERFECTLY!** 🎉

### ❌ If you see:
- Timeout after 120s
- 502 Bad Gateway

**→ Check:**
- Rollout status in dashboard
- Worker count in "Workers" tab
- Logs in "Logs" tab

---

## 🔄 Update Test URL Format

Based on your endpoint ID `46ifkgzb20xva0`, try these URLs:

1. Primary: `https://46ifkgzb20xva0.rp.runpod.net/serverless/invoke`
2. Alternative: `https://46ifkgzb20xva0.rp.runpod.ai/serverless/invoke`

(Use `.net` first as that's the standard RunPod domain)

---

**Need more help?** Check `DASHBOARD_TEST_URLS.md` for detailed guide.
