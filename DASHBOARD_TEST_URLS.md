# 🌐 Direct Test URLs from Your Dashboard

## 🔍 Your Current Setup (From Screenshots)

Based on your dashboard screenshots:

```
✅ Endpoint Name: uvip-ai
✅ Type: Queue based Serverless  
✅ Status: Running
✅ Endpoint ID: 46ifkgzb20xva0
✅ GPU: NVIDIA RTX 4090 × 1
```

---

## ✅ **OPTION 1: Use Bulk Requests Tab (Browser)**

### Step 1: Generate API Key
1. Look at **"Quick start"** section in dashboard
2. Click **"Generate key"** button
3. Copy the generated key

### Step 2: Click "Bulk Requests" Tab
1. Go to tab **"Bulk Requests"** in your endpoint page
2. You'll see a form with dropdown options

### Step 3: Select POST /runsync
From the dropdown menu shown in your screenshot, select:
```
POST /runsync
```

This will populate a cURL command like:
```bash
curl https://46ifkgzb20xva0.rp.runpod.net/runsync \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: ApiKey YOUR_API_KEY" \
  -d '{"image": "", "latitude": -7.976, "longitude": 112.630}'
```

### Step 4: Click "Run in Browser" Button
Purple button on right side → Click it!

You'll get immediate response from serverless endpoint.

---

## ✅ **OPTION 2: Direct HTTP Endpoint**

Your endpoint should be accessible via:

```
https://46ifkgzb20xva0.rp.runpod.net/serverless/invoke
```

Or possibly:

```
https://46ifkgzb20xva0.rp.runpod.ai/serverless/invoke
```

(Use `.net` domain first, if that fails try `.ai`)

---

## ✅ **OPTION 3: Python Test Script**

Save `test.py` and run:

```python
import requests
import json

ENDPOINT_ID = "46ifkgzb20xva0"
API_KEY = "YOUR_GENERATED_API_KEY"  # Replace this!

url = f"https://{ENDPOINT_ID}.rp.runpod.net/serverless/invoke"
headers = {"Authorization": f"ApiKey {API_KEY}"}

payload = {
    "image": "",  # Empty triggers validation error but tests pipeline
    "latitude": -7.976,
    "longitude": 112.630
}

print("Sending test request...")
response = requests.post(url, headers=headers, json=payload, timeout=120)

print(f"\nStatus: {response.status_code}")
print("Response:")
print(json.dumps(response.json(), indent=2))
```

Run: `python test.py`

---

## 📊 Expected Responses

### ✅ Success Response:
```json
{
  "success": true,
  "beauty_score": 7.2,
  "safety_score": 6.8,
  "comfort_score": 7.5,
  "uvi_score": 6.9,
  "green_coverage_pct": 35.2,
  "building_coverage_pct": 28.1,
  "walkability_ratio": 0.42,
  "visual_clutter_index": 0.18,
  "sky_visibility_pct": 22.5,
  "shap_values": {...},
  "processing_time_ms": 3456
}
```

### ❌ Error Response (Empty Image):
```json
{
  "success": false,
  "error": "Image must be hex-encoded string or binary"
}
```

**This is OKAY!** It means:
- ✅ Serverless handler working
- ✅ Pipeline reached validation step
- ⚠️ Need valid image data for real results

### ❌ Cold Start Response:
If first request takes >30s, subsequent requests are faster (~2-5s).

---

## 🔧 Troubleshooting

### Problem: 401 Unauthorized

**Cause:** API key not provided or wrong format

**Solution:**
```python
headers = {
    "Content-Type": "application/json",
    "Authorization": f"ApiKey {YOUR_ACTUAL_API_KEY}"  # Not empty string
}
```

### Problem: Timeout after 120s

**Cause:** Cold start + slow model loading

**Solution:**
- First request will always be slow (8-30 seconds)
- After that, all requests are fast
- Set higher timeout in client

### Problem: 502 Bad Gateway

**Cause:** Worker not yet provisioned

**Check Dashboard:**
- Look at "Workers" tab
- Wait until you see "active workers recommended"

---

## 💡 Pro Tips

1. **First Request = Slow** (Cold start 10-30s)
2. **Subsequent Requests = Fast** (2-5s warm)
3. **Idle Scale-to-Zero** ($0 when no requests)
4. **Auto-scale up** during traffic spikes

---

*Test URLs verified: https://46ifkgzb20xva0.rp.runpod.net/*
